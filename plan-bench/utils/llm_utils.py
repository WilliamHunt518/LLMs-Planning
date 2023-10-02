import asyncio

from transformers import StoppingCriteriaList, StoppingCriteria
import openai
import os
openai.api_key = "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"#os.environ["OPENAI_API_KEY"]
def generate_from_bloom(model, tokenizer, query, max_tokens):
    encoded_input = tokenizer(query, return_tensors='pt')
    stop = tokenizer("[PLAN END]", return_tensors='pt')
    stoplist = StoppingCriteriaList([stop])
    output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda(), max_new_tokens=max_tokens,
                                      temperature=0, top_p=1)
    return tokenizer.decode(output_sequences[0], skip_special_tokes=True)


def send_query(query, engine, max_tokens, model=None, stop="[STATEMENT]"):
    max_token_err_flag = False
    if engine == 'bloom':

        if model:
            response = generate_from_bloom(model['model'], model['tokenizer'], query, max_tokens)
            response = response.replace(query, '')
            resp_string = ""
            for line in response.split('\n'):
                if '[PLAN END]' in line:
                    break
                else:
                    resp_string += f'{line}\n'
            return resp_string
        else:
            assert model is not None
    elif engine == 'finetuned':
        if model:
            try:
                response = openai.Completion.create(
                    model=model['model'],
                    prompt=query,
                    temperature=0,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=["[PLAN END]"])
            except Exception as e:
                max_token_err_flag = True
                print("[-]: Failed GPT3 query execution: {}".format(e))
            text_response = response["choices"][0]["text"] if not max_token_err_flag else ""
            return text_response.strip()
        else:
            assert model is not None
    elif '_chat' in engine:
        
        eng = engine.split('_')[0]
        # print('chatmodels', eng)
        messages=[
        {"role": "system", "content": "You are the planner assistant who comes up with correct plans."},
        {"role": "user", "content": query}
        ]
        try:
            response = openai.ChatCompletion.create(model=eng, messages=messages, temperature=0)
        except Exception as e:
            max_token_err_flag = True
            print("[-]: Failed GPT3 query execution: {}".format(e))
        text_response = response['choices'][0]['message']['content'] if not max_token_err_flag else ""
        return text_response.strip()        
    else:
        try:
            response = openai.Completion.create(
                model=engine,
                prompt=query,
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop)
        except Exception as e:
            max_token_err_flag = True
            print("[-]: Failed GPT3 query execution: {}".format(e))

        text_response = response["choices"][0]["text"] if not max_token_err_flag else ""
        return text_response.strip()

def send_dialogue_query(query, engine, max_tokens, num_members, model=None, stop="[STATEMENT]"):
    num_members = 2
    max_token_err_flag = False
    eng = engine.split('_')[0]

    print("                                    (Q)")
    print(query[500:])

    conversation = [
        {},
    ]

    ## RUN A LONG RUN ON CURRENT SETTINGS< MIGHT BE GOOD :)
    #  Current is EQUAL for 50 runs

    # TODO try:
    #   Critic understand logic
    #   Critic is only given rules??
    sub_conversations = []
    sub_conversations.append([
        ## MAYBE this is causing it to argue with the example
       {"role": "system", "content": "You are a planner assistant who helps create correct plans. You only change the plan and never change the initial conditions."},#, and ignores incorrect feedback."},
        #{"role": "system", "content": "You are a planner assistant who comes up with plans that have one deliberate mistake (don't say what it is), and ignores incorrect feedback."},
        {"role": "user", "content": query+"\nNow let's discuss step by step what the next plan should be."}
    ])

    #response = openai.ChatCompletion.create(model=eng, messages=sub_conversations[0], temperature=0)
    #text_response = response['choices'][0]['message']['content'] if not max_token_err_flag else ""
    #sub_conversations[0].append({"role": "assistant", "content": text_response})

    sub_conversations.append([
        {"role": "system",
         "content": "You are a critic assistant who points out when my proposed actions aren't possible. You only change the plan and never change the initial conditions."},
        {"role": "user", "content": query + "\nNow let's discuss step by step what the next plan should be. Your main responsibilty is to point out when I say steps that aren't allowed within the possible actions."},
        # {"role": "user", "content": text_response+"\n\nImagine the initial conditions, then imagine performing each step and check that it follows the restrictions. Briefly explain any problems with my latest plan."}
    ])

    #sub_conversations.append([
    #    {"role": "system",
    #     "content": "You are a critic assistant who points out when I misunderstand the initial conditions. You only change the plan and never change the initial conditions."},
    #    {"role": "assistant",
    #     "content": query + "\nNow let's discuss step by step what the next plan should be. Your main responsibilty is to point out when I say wrong things about the initial conditions. Do not ciritcise my plan."},
    #    # {"role": "user", "content": text_response+"\n\nImagine the initial conditions, then imagine performing each step and check that it follows the restrictions. Briefly explain any problems with my latest plan."}
    #])

    names = ["planner", "ruleCritic"]#, "initCritic"]

    #print("                                    (0)")
    #print(names[0] + ": " + text_response)
    done = False
    for turn in range(0, 10):#range(1, 20):
        effective_index = turn % num_members
        message = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=sub_conversations[effective_index],
            max_tokens=600
        )['choices'][0]['message']['content']
        if turn == 1:
            message = message + "\nConsider this and let's argue about it."
        #message = trim_message(message)
        sub_conversations[effective_index].append({"role": "assistant", "content": message})
        [sub_con.append({"role": "user", "content": message}) for sub_con in sub_conversations if
         sub_con is not sub_conversations[effective_index]]
        print("                                    (" + str(effective_index) + ")")
        print(names[effective_index] + ": " + message)

        if done:
            final_response = sub_conversations[0][-1]
            res = final_response['content'].strip()
            return res
        elif ("REJECT" in message or "not possible" in message or "Not possible" in message or "ACCEPT" not in message
              or ["great day", "glad I could be of assistance", "fantastic day", "take care", "feel free to reach out"] in message):
            if turn > 6:
                # PROBABLY finshed, insert that question
                end_msg = "Ok I think we should wrap up this conversation. Please restate your plan as the final answer starting [PLAN]"
                sub_conversations[effective_index].append({"role": "assistant", "content": end_msg})
                print("                                    (" + str(effective_index) + ")")
                print(names[effective_index] + ": " + end_msg)
                [sub_con.append({"role": "user", "content": end_msg}) for sub_con in sub_conversations if
                 sub_con is not sub_conversations[effective_index]]
                done = True
        elif "ACCEPT" in message:
            effective_index = 2
            end_msg = "I think the plan is now correct. Please restate your plan as the final answer starting [PLAN]"
            sub_conversations[effective_index].append({"role": "assistant", "content": end_msg})
            print("                                    (" + str(effective_index) + ")")
            print(names[effective_index] + ": " + end_msg)
            [sub_con.append({"role": "user", "content": end_msg}) for sub_con in sub_conversations if
             sub_con is not sub_conversations[effective_index]]
            done = True


    final_response = sub_conversations[0][-2]
    res = final_response['content'].strip()
    return res


def safe_dialogue_query(query, engine, max_tokens, num_members, model=None, stop="[STATEMENT]", return_dict=None):
    res = send_dialogue_query(query, engine, max_tokens, num_members, model, stop)
    return_dict[0] = res
