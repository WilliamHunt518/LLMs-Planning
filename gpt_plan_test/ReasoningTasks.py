import os

import yaml
from Executor import Executor
from utils import *
from pathlib import Path
from tarski.io import PDDLReader
import argparse
import time

np.random.seed(42)

INTRO = """
I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do

Pick up a block
Unstack a block from on top of another block
Put down a block
Stack a block on top of another block

I have the following restrictions on my actions:
I can only pick up or unstack one block at a time.
I can only pick up or unstack a block if my hand is empty.
I can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.
I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.
I can only unstack a block from on top of another block if the block I am unstacking is clear.
Once I pick up or unstack a block, I am holding the block.
I can only put down a block that I am holding.
I can only stack a block on top of another block if I am holding the block being stacked.
I can only stack a block on top of another block if the block onto which I am stacking the block is clear.
Once I put down or stack a block, my hand becomes empty.
"""

INTRO_COST = """
I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do:

Pick up a block. It takes 1 minute to pick up a block.
Unstack a block from on top of another block. It takes 1 minute to unstack a block from on top of another block.
Put down a block. It takes 1 minute to put down a block.
Stack a block on top of another block. It takes 1 minute to stack a block on top of another block.

I have the following restrictions on my actions:
I can only pick up or unstack one block at a time.
I can only pick up or unstack a block if my hand is empty.
I can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.
I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.
I can only unstack a block from on top of another block if the block I am unstacking is clear.
Once I pick up or unstack a block, I am holding the block.
I can only put down a block that I am holding.
I can only stack a block on top of another block if I am holding the block being stacked.
I can only stack a block on top of another block if the block onto which I am stacking the block is clear.
Once I put down or stack a block, my hand becomes empty.
"""

success_template = "{} {} {} {}"
verbose_template="""
{}
--------- GPT3 response ---------
{}
--------- Extracted plan ---------
{}
-------- Ground truth plan ---------
{}
{}
"""

class ReasoningTasks():
    """
    Tasks:
    T1. Goal-directed reasoning
    T2. Paraphrasing of goals
    T3. Plan subset completion
    T4. Plan generalization
    T5. Optimality
    T6. Replanning
    T7. Plan execution
    """

    def __init__(self, engine, verbose=False):
        self.engine = engine
        self.verbose = verbose
        self.n_examples = 1
        self.max_gpt_response_length = 500

        self.plan_file = "sas_plan"
        self.gpt3_plan_file = "gpt_sas_plan"

    # ========================================== UTILS ========================================== #
    def compute_plan(self, domain, instance, timeout=30):
        fast_downward_path = os.getenv("FAST_DOWNWARD") 
        print(fast_downward_path)
        # Remove > /dev/null to see the output of fast-downward
        if '~' in fast_downward_path:
            assert os.path.exists(os.path.expanduser(fast_downward_path)+"fast-downward.py")
        else:
            assert os.path.exists(f"{fast_downward_path}/fast-downward.py")
        cmd = f"timeout {timeout}s {fast_downward_path}/fast-downward.py {domain} {instance} --search \"astar(lmcut())\" > /dev/null 2>&1"
        os.system(cmd)

        if not os.path.exists(self.plan_file):
            return ""
        return Path(self.plan_file).read_text()

    def read_config(self, config_file):
        with open(config_file, 'r') as file:
            self.data = yaml.safe_load(file)

    def get_problem(self, instance, domain):
        reader = PDDLReader(raise_on_error=True)
        reader.parse_domain(domain)
        return reader.parse_instance(instance)

    def get_executor(self, instance, domain):
        plan_executor = Executor(domain, instance)
        return plan_executor

    def save_output(self, output_file, final_output):
        os.makedirs(f"outputs/{self.engine}/", exist_ok=True)
        with open(f"outputs/{self.engine}/" + output_file + ".txt", 'w+') as f:
            f.write(final_output)
    # ========================================== TASKS ========================================== #
    def t1_t4(self, config_file, t1_or_t4="1_reasoning"):
        self.read_config(config_file)

        # ---- Uncomment the below lines to generate problem instances ---- #
        # for f_name in self.data['callbacks']:
        #     callback_obj = Callbacks(self.data)
        #     getattr(callback_obj, f_name)()
        # ---- Uncomment the above lines to generate problem instances ---- #

        domain_name = self.data['domain']
        domain_pddl = f'./instances/{self.data["domain_file"]}'
        instance_folder = f'./instances/{domain_name}/'
        instance = f'./instances/{domain_name}/{self.data["instances_template"]}'
        n_files = min(self.data['n_instances'], len(os.listdir(instance_folder)))

        final_output = ""
        correct_plans = 0
        for start in range(1, n_files + 2 - self.n_examples):
            query = INTRO
            for i in range(start, start + self.n_examples + 1):
                last_plan = True if i == start + self.n_examples else False
                get_plan = not last_plan
                cur_instance = instance.format(i)
                # --------------- Add to final output --------------- #
                final_output += f"\n Instance {cur_instance}\n"
                if self.verbose:
                    print(f"Instance {cur_instance}")
                # --------------- Read Instance --------------- #
                problem = self.get_problem(cur_instance, domain_pddl)
                # --------------------------------------------- #
                # ------------ Put plan and instance into text ------------ #
                gt_plan = self.compute_plan(domain_pddl, cur_instance)
                gt_plan_text = get_plan_as_text(self.data)
                query += fill_template(*instance_to_text_blocksworld(problem, get_plan, self.data))
                # --------------------------------------------------------- #

            # Querying GPT-3
            gpt3_response = send_query_gpt3(query, self.engine, self.max_gpt_response_length)

            # Do text_to_plan procedure
            _, gpt3_plan = text_to_plan_blocksworld(gpt3_response, problem.actions, self.gpt3_plan_file, self.data)
            # Apply VAL
            correct = int(validate_plan(domain_pddl, cur_instance, self.gpt3_plan_file))
            correct_plans += correct

            final_output += success_template.format('='*35, t1_or_t4, "SUCCESS" if correct else "FAILURE", '='*35)
            final_output += verbose_template.format(query, gpt3_response, gpt3_plan, gt_plan_text, '='*77) if self.verbose else ""
            if self.verbose: print(final_output)

            self.save_output("task" + t1_or_t4, final_output)

        os.remove(self.plan_file)
        os.remove(self.gpt3_plan_file)

        # --------------- Add to final output --------------- #
        final_output += f"[+]: The number of correct plans is {correct_plans}/{n_files}={correct_plans / (n_files) * 100}%"
        print(f"[+]: The number of correct plans is {correct_plans}/{n_files}={correct_plans / (n_files) * 100}%")
        self.save_output("task" + t1_or_t4, final_output)

    def t2_paraphrasing(self, config_file):
        def convert_state_to_text(state):
            text_list = []
            for i in state:
                pred = i.split('_')
                objs = [self.data["encoded_objects"][j] for j in pred[1:]]
                text_list.append(self.data['predicates'][pred[0]].format(*objs))

            state_text = text_list[0]
            if len(text_list) > 1:
                state_text = ", ".join(text_list[:-1]) + f" and {text_list[-1]}"

            return state_text

        def paraphrase_goal(exec):
            exec.complete_plan_execution()
            goal_state, full_state = list(exec.goal_state), list(exec.final_state)
            random.shuffle(goal_state)
            return len(goal_state), convert_state_to_text(full_state)

        self.read_config(config_file)
        start_time = time.time()
        domain_name = self.data['domain']
        domain = f'./instances/{self.data["domain_file"]}'
        instance = f'./instances/{domain_name}/{self.data["instances_template"]}'
        single_goal_instances = 0
        n = self.data['n_instances']
        skipped = 0
        corrects = {"Random": 0, "Full->Specific": 0, "Specific->Full": 0}
        final_output = ""
        for i in range(1, n + 1):
            cur_instance = instance.format(i)
            final_output += f"\n Instance {cur_instance}\n"
            plan_executor = self.get_executor(cur_instance, domain)

            problem = self.get_problem(cur_instance, domain)
            gt_plan = self.compute_plan(domain, cur_instance)

            if gt_plan == "":
                std_out = f"[-]: Timeout or error gathering Ground Truth plan for {cur_instance}. Continuing..."
                final_output += std_out
                print(std_out)
                skipped += 1
                continue
            gt_plan_text = get_plan_as_text(self.data)

            number_of_preds, goal_full = paraphrase_goal(plan_executor)

            try:
                init_specific, goal_specific, plan_specific = instance_to_text_blocksworld(problem, True, self.data)
                init_specific_shuffled, goal_specific_shuffled, _ = instance_to_text_blocksworld(problem, True,
                                                                                                 self.data,
                                                                                                 shuffle=True)
            except:
                final_output += f"[-]: Error parsing {cur_instance}. Skipping...\n"
                print(f"[-]: Error converting {cur_instance} to text. Continuing...")
                skipped += 1
                continue
            single_goal_instances += 1 if number_of_preds == 1 else 0
            # =============== Random =============== #
            query = INTRO
            query += fill_template(init_specific, goal_specific, plan_specific)
            query += fill_template(init_specific, goal_specific_shuffled, "")

            gpt3_response = send_query_gpt3(query, self.engine, self.max_gpt_response_length)
            _, gpt3_plan = text_to_plan_blocksworld(gpt3_response, problem.actions, self.gpt3_plan_file, self.data)

            correct = int(validate_plan(domain, cur_instance, self.gpt3_plan_file))
            corrects["Random"] += correct

            task_name = "(GOAL ORDERING CHANGE)"
            final_output += success_template.format('='*35, task_name, "SUCCESS" if correct else "FAILURE", '='*35)
            final_output += verbose_template.format(query, gpt3_response, gpt3_plan, gt_plan_text, '='*77) if self.verbose else ""
            if self.verbose: print(final_output)

            # =============== Full->Specific and Specific->Full =============== #
            descriptions = list(corrects.keys())[1:][::-1]
            for goal_1, goal_2, descr in zip([goal_specific, goal_full], [goal_full, goal_specific], descriptions):
                query = INTRO
                query += fill_template(init_specific, goal_1, plan_specific)
                query += fill_template(init_specific, goal_2, "")

                gpt3_response = send_query_gpt3(query, self.engine, self.max_gpt_response_length)
                _, gpt3_plan = text_to_plan_blocksworld(gpt3_response, problem.actions, self.gpt3_plan_file, self.data)
                correct = int(validate_plan(domain, cur_instance, self.gpt3_plan_file))
                corrects[descr] += correct

                final_output += success_template.format('='*35, descr, "SUCCESS" if correct else "FAILURE", '='*35)
                final_output += verbose_template.format(query, gpt3_response, gpt3_plan, gt_plan_text, '='*77) if self.verbose else ""
                if self.verbose: print(final_output)

            os.remove(self.plan_file)
            os.remove(self.gpt3_plan_file)

            exec_plans = i - skipped
            final_output += "\nResults:\n"
            print("Results")
            for k in corrects:
                final_output += f"{k} {corrects[k]}/{exec_plans} = {round(corrects[k] / exec_plans * 100, 2)}%"
                print(f"{k} {corrects[k]}/{exec_plans} = {round(corrects[k] / exec_plans * 100, 2)}%")
            final_output += "\n"
            final_output += f"Single goal instances: {single_goal_instances}\n"
            final_output += f"\nTotal time: {round(time.time() - start_time, 2)}s"
            os.makedirs(f"outputs/{self.engine}/", exist_ok=True)
            with open(f"outputs/{self.engine}/task2_paraphrase.txt", 'w+') as f:
                f.write(final_output)


        print(single_goal_instances)

    def t3_plan_subset(self, config_file):
        self.read_config(config_file)
        domain_name = self.data['domain']
        domain = f'./instances/{self.data["domain_file"]}'
        instance = f'./instances/{domain_name}/{self.data["instances_template"]}'
        n = self.data['n_instances']
        final_output = ""
        correct_plans = 0

        for i in range(494,495):
            cur_instance = instance.format(i)
            plan_executor = self.get_executor(cur_instance, domain)
            problem = self.get_problem(cur_instance, domain)
            final_output += f"\n Instance {cur_instance}\n"
            print(f"Instance {cur_instance}")
            # gt_plan = self.compute_plan(domain, cur_instance)
            full_plan, plan = generate_plan_subset(plan_executor, self.data, True)
            subset_plan, plan_subset = generate_plan_subset(plan_executor, self.data, False)
            gt_plan_text = get_plan_as_text(self.data, plan_subset)
            query = INTRO
            query += full_plan
            query += subset_plan
            # --------------------------------------------------------- #

            gpt3_response = send_query_gpt3(query, self.engine, self.max_gpt_response_length)
            _, gpt3_plan = text_to_plan_blocksworld(gpt3_response, problem.actions, self.gpt3_plan_file, self.data,
                                                    True)
            plan_executor.get_new_instance(change_goal=True, change_init=False)
            correct = int(validate_plan('pr-new-domain.pddl', 'pr-new-problem.pddl', self.gpt3_plan_file))

            task_name = ""
            succ_fail = "SUCCESS" if correct else "FAILURE"
            final_output += f"\n{'='*35} {task_name} {succ_fail} {'='*35}\n"
            correct_plans += correct
            final_output += f"{query}\n--------- GPT3 response ---------\n{gpt3_response}\n" + \
                            f"\n-------- Ground truth plan ---------\n{gt_plan_text}"
            final_output += f"\n{'='*77}\n"

            if self.verbose:
                print(f"{query}\n--------- GPT3 response ---------\n{gpt3_response}\n"
                      f"\n-------- Ground truth plan ---------\n{gt_plan_text}")
            self.save_output("task3_plan_subset", final_output)
            break

        exec_plans = n
        final_output += "\nResults:\n"
        final_output += f"No of correct plans, {correct_plans}/{exec_plans} = {round(correct_plans / exec_plans * 100, 2)}%"
        print(f"No of correct plans, {correct_plans}/{exec_plans} = {round(correct_plans / exec_plans * 100, 2)}%")
        try:
            os.remove(self.plan_file)
        except:
            pass
        os.remove(self.gpt3_plan_file)
        self.save_output("task3_plan_subset", final_output)

    def t5_optimality(self, config_file):
        self.read_config(config_file)
        domain_name = self.data['domain']
        domain = f'./instances/{self.data["domain_file"]}'
        instance = f'./instances/{domain_name}/{self.data["instances_template"]}'
        n = self.data['n_instances']
        final_output = ""
        correct_plans = 0
        for start in range(1, n + 2 - self.n_examples):
            query = INTRO_COST
            for i in range(start, start + self.n_examples + 1):
                last_plan = True if i == start + self.n_examples else False
                get_plan = not last_plan
                cur_instance = instance.format(i)
                plan_executor = self.get_executor(cur_instance, domain)
                final_output += f"\n Instance {cur_instance}\n"
                print(f"Instance {cur_instance}")
                # --------------- Read Instance --------------- #
                problem = self.get_problem(cur_instance, domain)
                # --------------------------------------------- #
                # ------------ Put plan and instance into text ------------ #
                gt_plan = self.compute_plan(domain, cur_instance)
                gt_plan_text = get_plan_as_text(self.data)
                inst, plan = optimality(plan_executor, self.data, get_plan)
                query += inst
                # --------------------------------------------------------- #

            # Querying GPT-3
            gpt3_response = send_query_gpt3(query, self.engine, self.max_gpt_response_length)

            # Do text_to_plan procedure
            _, gpt3_plan = text_to_plan_blocksworld(gpt3_response, problem.actions, self.gpt3_plan_file, self.data)
            # Apply VAL
            correct = int(validate_plan(domain, cur_instance, self.gpt3_plan_file))
            if not correct:
                final_output += f"{'='*35} FAILURE {'='*35}"
                final_output += '\n--------Invalid Plan-------\n'


            if self.verbose:
                print(f"{query}\n--------- GPT3 response ---------\n{gpt3_response}\n"
                      f"--------- Extracted plan ---------\n{gpt3_plan}"
                      f"\n-------- Ground truth plan ---------\n{gt_plan_text}")
                print(correct)

            if correct:
                cost = get_cost_gpt_3(gpt3_response)
                plan_list = [len(pl) > 0 for pl in gpt3_plan.split('\n')]
                actual_cost_gpt3 = sum(plan_list)
                txt= "---------WRONG COST OUTPUT BY LLM---------" if cost != actual_cost_gpt3 else "---------CORRECT COST OUTPUT BY LLM---------"
                print("COST OF GPT_3 PLAN", cost)
                if actual_cost_gpt3 == plan_executor.cost:
                    correct_plans += 1
                    final_output += f"{'='*35} SUCCESS {'='*35}"
                    final_output += '\n----------------Optimal Plan----------------\n'
                else:
                    final_output += f"{'='*35} FAILURE {'='*35}"
                    final_output += '\n----------------Sub-optimal Plan----------------\n'
                final_output += txt
            final_output += f"{query}\n--------- GPT3 response ---------\n{gpt3_response}\n" + \
                            f"--------- Extracted plan ---------\n{gpt3_plan}" + \
                            f"\n-------- Ground truth plan ---------\n{gt_plan_text}"
            final_output += f"\n{'='*77}\n"
            self.save_output("task5_optimality", final_output)
            break

        exec_plans = n
        final_output += f"No of correct plans, {correct_plans}/{exec_plans} = {round(correct_plans / exec_plans * 100, 2)}%"
        print(f"No of correct plans, {correct_plans}/{exec_plans} = {round(correct_plans / exec_plans * 100, 2)}%")
        try:
            os.remove(self.plan_file)
        except:
            pass
        os.remove(self.gpt3_plan_file)
        self.save_output("task5_optimality", final_output)

    def t6_replanning(self, config_file, harder):
        self.read_config(config_file)
        domain_name = self.data['domain']
        domain = f'./instances/{self.data["domain_file"]}'
        instance = f'./instances/{domain_name}/{self.data["instances_template"]}'
        n = self.data['n_instances']
        final_output = ""

        correct_plans = 0
        no_possible_plans = 0

        for start in range(1, n + 2 - self.n_examples):
            query = INTRO
            for i in range(start, start + self.n_examples + 1):
                last_plan = True if i == start + self.n_examples else False
                get_plan = not last_plan
                cur_instance = instance.format(i)
                plan_executor = self.get_executor(cur_instance, domain)
                problem = self.get_problem(cur_instance, domain)
                final_output += f"\n Instance {cur_instance}\n"
                print(f"Instance {cur_instance}")
                # gt_plan = self.compute_plan(domain, cur_instance)
                text, plan = replanning(plan_executor, self.data, get_plan, harder)
                query += text
            gpt3_response = send_query_gpt3(query, self.engine, self.max_gpt_response_length)
            gt_plan_text = get_plan_as_text(self.data, plan)
            # Do text_to_plan procedure
            _, gpt3_plan = text_to_plan_blocksworld(gpt3_response, problem.actions, self.gpt3_plan_file, self.data)
            # Apply VAL
            correct = int(validate_plan('pr-new-domain.pddl', 'pr-new-problem.pddl', self.gpt3_plan_file))

            correct_plans += correct
            succ_fail = "SUCCESS" if correct else "FAILURE"
            final_output += f"\n{'='*35} {succ_fail} {'='*35}\n"
            final_output += f"{query}\n--------- GPT3 response ---------\n{gpt3_response}\n" + \
                            f"--------- Extracted plan ---------\n{gpt3_plan}" + \
                            f"\n-------- Ground truth plan ---------\n{gt_plan_text}"
            final_output += f"\n{'='*77}\n"

            if self.verbose:
                print(f"{query}\n--------- GPT3 response ---------\n{gpt3_response}\n"
                      f"--------- Extracted plan ---------\n{gpt3_plan}"
                      f"\n-------- Ground truth plan ---------\n{gt_plan_text}")
                print(correct)
            self.save_output("task6_replanning", final_output)
            break


        exec_plans = n - no_possible_plans
        final_output += f"No of correct plans, {correct_plans}/{exec_plans} = {round(correct_plans / exec_plans * 100, 2)}%"
        print(f"No of correct plans, {correct_plans}/{exec_plans} = {round(correct_plans / exec_plans * 100, 2)}%")
        try:
            os.remove(self.plan_file)
        except:
            pass
        hard_or_easy = "hard" if harder else "easy"
        try:
            os.remove(self.gpt3_plan_file)
        except:
            pass
        self.save_output("task6_replanning", final_output)

    def t7_plan_execution(self, config_file):
        self.read_config(config_file)
        domain_name = self.data['domain']
        domain = f'./instances/{self.data["domain_file"]}'
        instance = f'./instances/{domain_name}/{self.data["instances_template"]}'
        n = self.data['n_instances']
        final_output = ""
        correct_answers = 0
        for start in range(1, n + 2 - self.n_examples):
            query = INTRO
            for i in range(start, start + self.n_examples + 1):
                last_plan = True if i == start + self.n_examples else False
                get_plan = not last_plan
                cur_instance = instance.format(i)
                plan_executor = self.get_executor(cur_instance, domain)
                final_output += f"\n Instance {cur_instance}\n"
                print(f"Instance {cur_instance}")
                # --------------- Read Instance --------------- #
                inst, answer = plan_execution(plan_executor, self.data, get_plan)
                query += inst
                # --------------------------------------------------------- #

            # Querying GPT-3
            gpt3_response = send_query_gpt3(query, self.engine, self.max_gpt_response_length)

            # Do text_to_plan procedure
            # gpt3_plan = text_to_plan_blocksworld(gpt3_response, problem.actions, self.gpt3_plan_file, self.data)

            # Apply VAL
            correct = gpt3_response.strip() == answer.strip()
            correct_answers += correct
            succ_fail = "SUCCESS" if correct else "FAILURE"
            final_output += f"\n{'='*35} {succ_fail} {'='*35}\n"
            final_output += f"{query}\n--------- GPT3 response ---------\n{gpt3_response}\n" + \
                            f"\n-------- Ground truth answer ---------\n{answer}"
            if self.verbose:
                print(f"{query}\n--------- GPT3 response ---------\n{gpt3_response}\n"
                      # f"--------- Extracted plan ---------\n{gpt3_plan}"
                      f"\n-------- Ground truth answer ---------{answer}")
                print(correct)
            final_output += f"\n{'='*77}\n"

            self.save_output("task7_plan_execution", final_output)
            break

        exec_plans = n
        final_output += f"No of correct plans, {correct_answers}/{exec_plans} = {round(correct_answers / exec_plans * 100, 2)}%"
        print(f"No of correct plans, {correct_answers}/{exec_plans} = {round(correct_answers / exec_plans * 100, 2)}%")
        try:
            os.remove(self.plan_file)
        except:
            pass
        self.save_output("task7_plan_execution", final_output)


if __name__ == '__main__':
    random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='t1', help='Task to run \
    \n t1 = Goal Directed Reasoning\
    \n t2 = Goal Reformulation \
    \n t3 = Plan Reuse \
    \n t4 = Plan Generalization\
    \n t5 = Optimal Planning \
    \n t6 = Replanning (easier) \
    \n t7 = Plan Execution \
    ')
    parser.add_argument('--engine', type=str, default='davinci', help='Engine to use')
    parser.add_argument('--verbose', type=str, default="False", help='Verbose')
    args = parser.parse_args()
    task = args.task
    engine = args.engine
    verbose = eval(args.verbose)
    tasks_obj = ReasoningTasks(engine, verbose)
    if task == 't1':
        config_file = './configs/t1_goal_directed_reasoning.yaml'
        tasks_obj.t1_t4(config_file)
    elif task == 't2':
        config_file = './configs/t2_paraphrasing.yaml'
        tasks_obj.t2_paraphrasing(config_file)
    elif task == 't3':
        config_file = './configs/t3_plan_subset.yaml'
        tasks_obj.t3_plan_subset(config_file)
    elif task == 't4':
        config_file = './configs/t4_plan_generalization.yaml'
        tasks_obj.t1_t4(config_file, "4_generalization")
    elif task == 't5':
        config_file = './configs/t3_plan_subset.yaml'
        tasks_obj.t5_optimality(config_file)
    elif task == 't6':
        config_file = './configs/t3_plan_subset.yaml'
        tasks_obj.t6_replanning(config_file, harder=0)
    # tasks_obj.t6_replanning(config_file, harder=0)
    elif task == 't7':
        config_file = './configs/t3_plan_subset.yaml'
        tasks_obj.t7_plan_execution(config_file)
