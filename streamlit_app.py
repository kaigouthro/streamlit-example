#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import re
import tokenizers
import openai
import datasets
from datasets import load_dataset

import streamlit as st
from streamlit_option_menu import option_menu

st_state = st.session_state

# API_KEY       : str = st.text_input("API",key="api_key")
st.session_state.api_key = st.session_state.get("api_key", None)


PRESETFOLDER = "./presets"
OUTPUTFOLDER = "./output"

# st.set_page_config(layout="wide")


from dataclasses import dataclass
from pathlib import Path


# This class represents a simple database that stores its data as files in a directory.
class DB:
    """A simple key-value store, where keys are filenames and values are file contents."""

    def __init__(self, path):
        self.path = Path(path).absolute()

        self.path.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, key):
        full_path = self.path / key

        if not full_path.is_file():
            raise KeyError(key)
        with full_path.open("r", encoding="utf-8") as f:
            return f.read()

    def __setitem__(self, key, val):
        full_path = self.path / key
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(val, str):
            full_path.write_text(val, encoding="utf-8")
        else:
            # If val is neither a string nor bytes, raise an error.
            raise TypeError("val must be either a str or bytes")



default_templates = {
    "EXAMPLE": {
        "system_basis": "You are a <NON_TERSE> who (is, can, knows about, is skilled at...) <NON_TERSE>.",
        "instructions": "Do <NON_TERSE> to the data",
        "context"     : "The current topic's <NON_TERSE> additional information that you may need to consider.",
    },
    "Custom": {
        "system_basis": "YOU....",
        "instructions": "Do.... ",
        "context"     : "information that you may need to consider.",
    },
    "Meeting": {
        "system_basis": "You are a professional Virtual Assistant who helps with scheduling and managing tasks.",
        "instructions": "Schedule a meeting with the following details:",
        "context"     : "Keep in mind the time zone, date, time, duration, and locations, and make sure it is saved.",
    },
    "Author": {
        "system_basis": "You are a creative Storyteller who is about to write a thrilling mystery novel.",
        "instructions": "Write a new chapter in the story with these events",
        "context"     : "Detective Jones is always on the verge of solving the case, but something hilarious happens...",
    },
    "Travel_Guide": {
        "system_basis": "You are a knowledgeable Tour Guide who provides information about famous landmarks and historical sites.",
        "instructions": "Describe the significance and architectural style of the Taj Mahal.",
        "context"     : "The Taj Mahal is an iconic monument located in Agra, India. It was built by the Mughal emperor Shah Jahan in memory of his wife...",
    },
    "Chef": {
        "system_basis": "You are a skilled Recipe Developer who specializes in creating delicious and healthy meals.",
        "instructions": "Create a recipe for a vegan-friendly salad using seasonal ingredients.",
        "context"     : "Spring is known for its abundance of fresh produce, such as...",
    },
    "Investor": {
        "system_basis": "You are a seasoned Financial Advisor who assists clients with investment strategies.",
        "instructions": "Recommend an investment portfolio for a risk-averse individual nearing retirement.",
        "context"     : "The client has a moderate amount of savings and wants to ensure steady growth while minimizing potential losses.",
    },
}

INSTRUCTIONS = {
    "EXAMPLE"                   : "Find a way to make this work, and share how we can improve it.",
    "fix_code_erros"            : "Fix any errors you find in the code",
    "translate_to_english"      : "Translate this code to English",
    "find_humor"                : "Look for a way to see this as humorous and lighthearted",
    "explain_via_metaphor"      : "Explain this code using an illustrative metaphor or simile",
    "generate_poem"             : "Generate a poem on the theme of <NON_TERSE>",
    "create_mind_map"           : "Create a mind map illustrating the different components of <NON_TERSE>",
    "implement_algorithm"       : "Implement <NON_TERSE> algorithm in the programming language of your choice",
    "identify_common_patterns"  : "Identify common patterns in the data and propose ways to exploit them",
    "predict_future_trends"     : "Predict future trends based on the available data and explain your reasoning",
    "optimize_performance"      : "Optimize the performance of <NON_TERSE> by identifying bottlenecks and suggesting improvements",
    "perform_sentiment_analysis": "Perform sentiment analysis on a given text and provide an overall sentiment score",
}

CONTEXT = {
    "EXAMPLE"              : "Use mathematics and algorithms to help",
    "algorithms"           : "My knowledge of Algorithms is broad, but not extensively understood",
    "humor"                : "We get along better with humor to help",
    "metaphor"             : "Using metaphors helps us talk",
    "context_free_grammar" : "Use EBNF, ANTLR4, and the W3C-EBNF XML specification to help",
    "data_analysis"        : "Expertise in data analysis helps",
    "business_strategy"    : "Apply our understanding of business strategy to the given task",
    "machine_learning"     : "Utilize machine learning techniques to tackle the problem",
    "ethics"               : "Consider ethical implications and provide solutions that align with ethical principles",
    "cybersecurity"        : "Incorporate cybersecurity measures into the solution to mitigate potential risks",
    "personalization"      : "Tailor the solution to individual preferences and needs",
    "sustainable_solutions": "Ensure the solutions offered are environmentally sustainable",
}
SYSTEM_BASIS = {
    "EXAMPLE"                        : "You Like Solving Problems",
    "researcher"                     : "You are a knowledgeable Researcher who specializes in gathering and analyzing data for academic studies and reports.",
    "graphic_designer"               : "You are a talented Graphic Designer who creates visually stunning designs for various digital and print media.",
    "customer_support_representative": "You are a dedicated Customer Support Representative who assists customers with their inquiries, concerns, and technical issues.",
    "software_engineer"              : "You are a skilled Software Engineer who develops innovative and efficient software solutions for various industries.",
    "marketing_strategist"           : "You are an experienced Marketing Strategist who devises effective marketing campaigns to promote products and services.",
    "data_scientist"                 : "You are a proficient Data Scientist who analyzes large datasets and derives valuable insights to drive business decisions.",
    "ux_ui_designer"                 : "You are a creative UX/UI Designer who designs intuitive and user-friendly interfaces for websites and applications.",
    "quality_assurance_analyst"      : "You are a meticulous Quality Assurance Analyst who ensures the accuracy and functionality of software products before they are released.",
    "project_manager"                : "You are a skilled Project Manager who oversees the planning, execution, and delivery of complex projects within given timeframes and budgets.",
    "copywriter"                     : "You are a talented Copywriter who creates compelling and persuasive content for marketing materials, advertisements, and websites.",
}


# 5. Add on_change callback
def on_change(key):
    selection = st_state[key]
    st.write(selection)


def get_api_key(st_state):
    """get api key if none is stored in session state"""
    api_key = " "
    if st.session_state.api_key:
        api_key =  st.session_state.api_key
        if api_key == st_state["api_key"] :
            return api_key
        if api_key == st.text_input("Enter your API key"):
            st_state["api_key"]  = api_key
            return api_key
    else:
        api_key = st_state["api_key"] = st.text_input("Enter your API key")
    return api_key



def get_models():
    """get modelsavailable from openai"""
    resp      = openai.Model.list(API_KEY)
    model_ids = [model["id"] for model in resp["data"]]
    model_ids = [model for model in model_ids if "gpt" in model]
    model_ids.sort()
    return model_ids


def sidebar_options():
    openai.api_key          = get_api_key(st_state)
    st_state["model"]       = st.sidebar.selectbox("Completion Engine", get_models())
    st_state["temperature"] = st.sidebar.slider(
        "Temperature:", 0.0, 1.0, 0.2, step = 0.01
    )
    st_state["top_p"]     = st.sidebar.slider("Top P:", 0.0, 1.0, 1.0, step=0.01)
    model                 = st_state["model"]
    temperature           = st_state["temperature"]
    top_p                 = st_state["top_p"]
    st_state["maxtokens"] = {
        "gpt-3.5-turbo"         : 4096,
        "gpt-3.5-turbo-16k"     : 16384,
        "gpt-3.5-turbo-0613"    : 4096,
        "gpt-3.5-turbo-16k-0613": 16384,
        "text-davinci-003"      : 4097,
        "text-davinci-002"      : 4097,
        "code-davinci-002"      : 8001,
    }
    model_max = st_state["maxtokens"][
        f'{model if model in st_state.maxtokens else "gpt-3.5-turbo-0613"}'
    ]
    st_state["maximum_tokens"] = st.sidebar.slider(
        "Max Tokens:", min_value = 32, max_value = model_max, value = 1024, step = 16
    )
    st_state["generate_all"] = st.sidebar.button("Generate All")


class Initializer:
    """set up the state for presets"""
    def __init__(self, state):
        self.initialized          = None
        self.state                = state
        self.state["presets"]     = {"system_basis": {}, "instructions": {}, "context": {}}
        self.state["edit_preset"] = {
            "local"  : "",
            "global" : "system_basis",
            "content": "",
        }
        self.state["prompts"] : {"response": " '"} = {}
        self.state["template"]  = ""
        self.state["templates"] = {"template": default_templates}
        self.init(state)
        self.main()

    def init(self, st_state):
        """set main config and presets information to state"""
        is_installed            = None
        st_state["initialized"] = None
        if st.session_state["initialized"] is None:
            is_installed = False
        if is_installed is None:
            st_state.initialized = False
            return
        is_installed = st_state["initialized"]
        try:
            with open("main.json", "r") as f:
                configs                    = json.load(f)
                is_installed               = configs["installed"]
                st_state["default_prompt"] = configs["default_prompt"]

        except Exception:
            if not os.path.exists("configs.json"):
                with open("configs.json", "w") as f:
                    configs = {"installed": True, "default_prompt": "Custom"}
                    json.dump(configs, f, indent=4)
                    is_installed               = configs["installed"]
                    st_state["default_prompt"] = configs["default_prompt"]

    def main(self):
        """main function to craete the presets if they do not exist"""

        if not os.path.exists(f"{PRESETFOLDER}"):
            os.mkdir(f"{PRESETFOLDER}")


        # setup
        self.templates()

        # create the folders and json files
        if not os.path.exists(f"{PRESETFOLDER}/system_basis.json"):
            self.write_basis()
        if not os.path.exists(f"{PRESETFOLDER}/instructions.json"):
            self.write_instructions()
        if not os.path.exists(f"{PRESETFOLDER}/contexts.json"):
            self.write_contexts()
        self.state["presets"]["system_basis"] = json.loads(
            open(f"{PRESETFOLDER}/system_basis.json", "r", encoding="utf-8").read()
        )
        self.state["presets"]["instructions"] = json.loads(
            open(f"{PRESETFOLDER}/instructions.json", "r", encoding="utf-8").read()
        )
        self.state["presets"]["context"] = json.loads(
            open(f"{PRESETFOLDER}/contexts.json", "r", encoding="utf-8").read()
        )
        self.state["system_basis"] = st_state["presets"]["system_basis"]
        self.state["instructions"] = st_state["presets"]["instructions"]
        self.state["context"]      = st_state["presets"]["context"]
        self.state["explanations"] = {
            "system_basis": "This is what the AI will try to be. It's like a self-identification, or a self-description.",
            "instructions": "This is what the AI will attempt to do for each of the prompts being submitted. It's like a global mandate for all items, or a specific goal that each and every prompt will attempt to accomplish.",
            "context"     : "This is the context that the AI will use to generate the text. It's where you place knowledge to reference, a global context for all prompts, or something that contains a dictionar or ruleset to reference .",
        }

    def write_basis(self):
        basis = SYSTEM_BASIS
        open(f"{PRESETFOLDER}/system_basis.json", "w", encoding="utf-8").write(
            json.dumps(basis, ensure_ascii=False, indent=4)
        )

    def write_instructions(self):
        instructions = INSTRUCTIONS
        open(f"{PRESETFOLDER}/instructions.json", "w", encoding="utf-8").write(
            json.dumps(instructions, ensure_ascii=False, indent=4)
        )

    def write_contexts(self):
        contexts = CONTEXT
        open(f"{PRESETFOLDER}/contexts.json", "w", encoding="utf-8").write(
            json.dumps(contexts, ensure_ascii=False, indent=4)
        )

    def templates(self):
        if not os.path.exists(f"{PRESETFOLDER}"):
            os.mkdir(f"{PRESETFOLDER}")
        if not os.path.exists(f"{PRESETFOLDER}/templates.json"):
            open(f"{PRESETFOLDER}/templates.json", "w", encoding="utf-8").write(
                json.dumps(default_templates, ensure_ascii=False, indent=4)
            )
        self.state["templates"] = default_templates
        self.state["template"]  = st_state.templates["EXAMPLE"]
        self.state["presets"]   = {"system_basis": {}, "instructions": {}, "context": {}}


def getHeading(preset: str = None):
    heading: str = (
        "What Do i DO?"
        if preset == "system_basis"
        else "My Job is to"
        if preset == "instructions"
        else "My Context is:"
        if preset == "instructions"
        else " "
    )
    return heading


def update_preset_display(st_state, preset, local_preset):
    heading             = getHeading(preset)
    current_setting     = st_state["presets"][preset]
    template_is_example = st_state["template"] == "EXAMPLE"
    preset_is_example   = current_setting      == "EXAMPLE"
    displayed           = st_state["presets"][preset][local_preset]

    if template_is_example or preset_is_example:
        st.markdown("# **`Example is not editable`**")

    if (
        md_text := f"""
        | SETTING         | **{local_preset}**                           |
        | --------------- | -------------------------------------------  |
        | `Editing`       | {preset}                                   |
        | `What is this?` | {st_state['explanations'][preset]}         |
        | `AI Mindset `   | {heading}                                  |
        | `example`       | {st_state['templates']['EXAMPLE'][preset]} |
        """
    ):
        st.markdown(md_text)

    updated_content = st.text_area("", displayed, key=f"{preset}_edit")

    if updated_content != displayed:
        st_state["edit_preset"]["content"] = updated_content
        # Update the preset in the presets dictionary
        st_state["presets"][preset][local_preset] = updated_content

        # Save the presets to the JSON file
        with open(f"./presets/{preset}.json", "w", encoding="utf-8") as f:
            json.dump(st_state["presets"][preset], f, ensure_ascii=False, indent=4)


def preset_display(st_state):
    c0, c1, c2 = st.columns([0.2, 0.65, 0.15])
    st.write("Look at:")
    c0.markdown("### `Template`")
    c0.markdown("### `Setting`")
    c0.markdown("### `Preset`")
    cl1, cl2, cl3 = st.columns(3)

    if template := c1.selectbox(
        "Template", list(st_state["templates"].keys()), label_visibility = "collapsed"
    ):
        st_state["template"]        = template
        st_state["prompt_settings"] = template
    # """

    # The option_menu function accepts the following parameters:

    # menu_title (required): the title of the menu; pass None to hide the title
    # options (required): list of (string) options to display in the menu; set an option to "---" if you want to insert a section separator
    # default_index (optional, default=0): the index of the selected option by default
    # menu_icon (optional, default="menu-up"): name of the bootstrap-icon to be used for the menu title
    # icons (optional, default=["caret-right"]): list of bootstrap-icon names to be used for each option; its length should be equal to the length of options
    # orientation (optional, default="vertical"): "vertical" or "horizontal"; whether to display the menu vertically or horizontally
    # styles (optional, default=None): A dictionary containing the CSS definitions for most HTML elements in the menu, including:
    # "container": the container div of the entire menu
    # "menu-title": the <a> element containing the menu title
    # "menu-icon": the icon next to the menu title
    # "nav": the <ul> containing "nav-link"
    # "nav-item": the <li> element containing "nav-link"
    # "nav-link": the <a> element containing the text of each option
    # "nav-link-selected": the <a> element containing the text of the selected option
    # "icon": the icon next to each option
    # "separator": the <hr> element separating the options
    # manual_select: Pass to manually change the menu item selection. The function returns the (string) option currently selected
    # on_change: A callback that will happen when the selection changes. The callback function should accept one argument "key". You can use it to fetch the value of the menu (see example 5)
    # """

    #preset = option_menu()
    preset = st.radio(
        "Setting to Edit",
        ["system_basis", "instructions", "context"],
        label_visibility = "collapsed"
    )

    local_preset = c1.radio(
        "presets",
        list(st_state["presets"][preset].keys()),
        label_visibility = "collapsed",
    )

    if template  != "EXAMPLE":
        add_new   =  c2.button("Add", key="Addbutton")
        n_name    =  f"new {preset}"
        new_name  =  True
        if preset in st_state["presets"]:
            edit      =  c2.button("Edit", key="editAddbutton")
            save_this =  c2.button("Save", key="Savebutton")

            if template  == "Custom" and (edit or add_new):
                new_name = False

            n_name = local_preset if edit else n_name

            name   = st.text_input(
                "name", n_name, disabled = new_name, on_change = None, key = "newname"
            )
            st_state["presets"][preset][name] = st_state["edit_preset"]["content"]

            if save_this:
                # add the preset to the json file
                open(f"./presets/{preset}.json", "w", encoding="utf-8").write(
                    json.dumps(st_state["presets"][preset], ensure_ascii=False, indent=4)
                )
                new_name = True

    update_preset_display(st_state, preset, local_preset)


def prompt_process(i):
    st_state[f"start_time{i}"] = time.time()
    msg_content               = st_state["prompts"][f"prompt{i}"]["msg_content"]
    completion                = openai.ChatCompletion.create(
        model    = st_state["model"],
        messages = [
            {"role": "system", "content": st_state["system_basis"]},
            {
                "role"   : "assistant",
                "content": "the info you have provided so far is:"
                + st_state["context"],
            },
            {
                "role"   : "user",
                "content": "Thanks, here is the instructions on what i will need you to do : "
                + st_state["instructions"],
            },
            {
                "role"   : "assistant",
                "content": "Ok, please send the data to be processed and i will perform the actions. you can say 'SUMMARIZE' keyword to start if you would like me to respond instead with a brief summary of what i would be doing, 'SUGGEST' if you would prefer my 3 best outside the box ideas on how you could achieve a better result or an alternative goal outcome",
            },
            {
                "role"   : "user",
                "content": f"Here is what is to be processed{msg_content}",
            },
        ],
        temperature = st_state["temperature"],
        max_tokens  = st_state["maximum_tokens"],
        top_p       = st_state["top_p"],
    )

    prompt_obj                    = st_state['prompts'][f'prompt{i}']
    prompt_obj['response']        = completion.choices[0]["message"]["content"]
    response                      = prompt_obj['response']
    prompt_tokens                 = completion['usage']["prompt_tokens"]
    response_tokens               = completion['usage']["completion_tokens"]
    total_tokens                  = completion['usage']["total_tokens"]
    word_count                    = len(re.findall(r"\w+", response))
    prompt_obj[f"resp{i}"]        = response
    prompt_obj["response"]        = response
    prompt_obj["prompt_tokens"]   = prompt_tokens
    prompt_obj["response_tokens"] = response_tokens
    prompt_obj["word_count"]      = word_count
    prompt_obj["time taken"]      = (
        time.time() - st_state[f"start_time{i}"]
    )
    prompt_obj["full"] = completion
    return response


def generate_response(i, msg_content: str = None):
    try:
        prompt_process(i)
    except Exception as e:
        st_state["prompts"][f"prompt{i}"]["response"] = str(e)
        return str(e)
    return True


def delayed_completion(delay_in_seconds: float = 1, i: int = 0):
    """Delay a completion by a specified amount of time."""
    msg_content = st_state["prompts"][f"prompt{i}"]["msg_content"]
    # Sleep for the delay
    time.sleep(delay_in_seconds)
    rep    = prompt_process(i)
    timeis = time.time().__str__().replace(".", "_")
    if rep:
        with open(f"train_item_{i}_{timeis}.txt", "w", encoding="utf-8") as f:
            f.write(rep)
    st_state["prompts"][f"prompt{i}"][
        "response"
    ] = rep  # Update the response for index i
    st.session_state["prompts"] = st_state[
        "prompts"
    ]  # Update the session state with the updated prompts
    return rep


import tiktoken
encodings = tiktoken.list_encoding_names()
encodings.reverse()

def tokencount(string: str = "Buffer") -> int:
    """Returns the number of tokens in a text string."""
    tokens = tiktoken.get_encoding("cl100k_base").encode(string or "buffer")
    return len(tokens)


def display_inputs(st_state, i):
    init_prompt(st_state, i)
    tis_prompt              = st_state['prompts'][f"prompt{i}"]
    tis_prompt['container'] = st.container()
    container               = tis_prompt['container']
    col2, col1              = container.columns([0.65, 0.35])

    with container.container():

        tis_prompt["msg_content"] = col2.text_area(
            f"Item To Prompt With : Row `{i+1}` of `{st_state.row_count}`",
            value  = f'{tis_prompt["msg_content"]}',
            height = 100,
            key    = f"msg_content{i}",
            )

        tokens = tokencount(tis_prompt["msg_content"])

        with col1.container():
            ic2, ic1 = st.columns(2)
            ic1.markdown(f"Tokens Free `{st_state['maxtokens'][st_state.model] - tokens}`")
            ic2.text(
                f' Notes{tis_prompt["notes"] if st_state.show_notes else ""}'
            )

            ic1.markdown(
            f"""
            | Stat     | Val                                     |
            | -------- | --------------------------------------- |
            | Sent     | {tis_prompt["prompt_tokens"]}    |
            | Received | {tis_prompt["response_tokens"]}   |
            | Wrds     | {tis_prompt["word_count"]}        |

            """
        )

    resp  = st_state[f"resp{i}"] in st_state
    rep   = st_state[f"resp{i}"] if resp else None

    button_container = col2.container()
    with button_container:
        lbut, rbut = st.columns(2)
        if lbut.button( f"Generate Response {i+1}",
            key  = f"generate_button_{i}",
            help = "Generate a response for this prompt", ):
            resp = True
        box = DB(f"Box {i}")


        rbut.download_button(f'Download OpenAI Response {i}',json.dumps(st_state["prompts"][f"prompt{i}"]["full"] ))


    if resp or st_state.generate_all:
        rep = delayed_completion(0.1, i)
        st_state[f"resp{i}"] = rep
        resp = False

    f"resp{i}"

def main(st_state):

    st.sidebar.title("Settings")
    sidebar_options()

    show_notes             = st.sidebar.checkbox("Show Notes", value="TRUE")
    st_state["show_notes"] = show_notes
    st_state["row_count"]  = 0
    st_state["status"]     = {"message": ""}

    # for dataset editing/translating purposes
    use_dataset = st.sidebar.checkbox("Use Dataset", False)
    dataset     = None
    if use_dataset:
        datasetname        = st_state.get("dataset", "OllieStanley/humaneval-mbpp-codegen-qa")
        dataset            = datasets.load_dataset(datasetname, keep_in_memory=True)
        st_state.row_count = len(dataset["train"])

    #st.sidebar.write("System Message:", st_state.status["msg_content"])

    side_column1, side_column2 = st.sidebar.columns(2)

    st_state["row_count"] = side_column1.number_input(
        "Prompts",
        min_value = 1,
        max_value = 10,
        value     = max(1, st_state["row_count"] ),
        step      = 1,
    )

    pre_settings  = st_state["presets"]

    st_state["system_basis"] = st.sidebar.selectbox(
        "Set system_basis", pre_settings["system_basis"], key = "setsystem_basis"
    )
    st_state["instructions"] = st.sidebar.selectbox(
        "Set instructions", pre_settings["instructions"], key = "setinstructions"
    )
    st_state["context"] = st.sidebar.selectbox(
        "Set context", pre_settings["context"], key = "setcontext"
    )


    s_setting = st_state["system_basis"]
    i_setting = st_state["instructions"]
    c_setting = st_state["context"]

    confirmed = st.sidebar.checkbox("Confirm Reset", value=False)


    st.title     ("GPT-CHAT Mass Prompter")
    st.subheader (f"i am a: `{ s_setting } `"     )
    st.markdown  (f" {st_state['presets']['system_basis'][s_setting].replace('You are','I am')}")
    st.subheader (f"My Goal for each: `{ i_setting } `"     )
    st.markdown  (f" {st_state['presets']['instructions'][i_setting]}")

    if side_column1.button("Reset Prompts") and confirmed:
        st_state.row_count = 1

    prompt_state = st_state["prompts"]

    if st_state.row_count and st_state.row_count != len(prompt_state):
        for i in range(st_state.row_count):
            if i > len(prompt_state) - 1:
                new_prompt = {
                    "notes"          : None,
                    "msg_content"    : None,
                    "response"       : None,
                    "prompt_tokens"  : None,
                    "response_tokens": None,
                    "word_count"     : None,
                    "full"           : None,
                    "time takekn"    : None,
                }
                prompt_state[f"prompt{i}"] = new_prompt
                st_state[f"resp{i}"]       = " "
            elif len(prompt_state) - i < 0:
                # remove key
                del prompt_state[f"prompt{i}"]
    st.divider()
    if st.sidebar.checkbox("Show Advanced Options"):
        preset_display(st_state)
    else:
        # show presets (make context as collapsible)
        st.sidebar.markdown(st_state["context"])

        # for i in range(st_state["row_count"]):

        #     init_prompt(st_state, i)
        # container = st.container()
        # col1,      col2 = container.columns([.35,.65])
        #     # ...
        # st_state['prompts'][f'prompt{i}']['do_generate_response'] = False
        # with st.container():
        #         col1.markdown(f" ### `Row {i+1} of {st_state.row_count}`")
        # if col1.button(f"Generate Response {i+1}", key=f"generate_button_{i}", help="Generate a response for this prompt"):
        # st_state['prompts'][f'prompt{i}']['do_generate_response'] = True

        #         msg_content = st.text_area(
        #             "Item To Prompt With",
        # value  = f'{st_state["prompts"][f"prompt{i}"]["msg_content"]}',
        # height = 100,
        # key    = f"msg_content{i}"
        #         )
        # tokens = tokencount(msg_content)
        # with col1.container():
        # ic1, ic2 = st.columns(2)
        #             ic1.markdown(f'|`Remaining`|\n|-|')
        #             ic2.markdown(f"|`{ st_state['maxtokens'][st_state.model] - tokens }`|\n|-|")
        #         col2.markdown(f"""
        #             | Stat     | Val                                     |
        #             | -------- | --------------------------------------- |
        #             | Sent     | { st_state["prompts"][f"prompt{i}"]["prompt_tokens"]}    |
        #             | Received | {st_state["prompts"][f"prompt{i}"]["response_tokens"]}   |
        #             | Wrds     | {st_state["prompts"][f"prompt{i}"]["word_count"]}        |
        #             | Wrds     | {st_state["prompts"][f"prompt{i}"]["word_count"]}        |
        #             """)
        #         col2.text(f'{st_state["prompts"][f"prompt{i}"]["notes"] if show_notes else ""}')
        # st_state["prompts"][f"prompt{i}"]["RUN"] = st_state['prompts'][f'prompt{i}']['do_generate_response']

        # if st_state['prompts'][f'prompt{i}']['do_generate_response']:
        # rep                                                       = delayed_completion(0.1, i, msg_content)
        # st_state['prompts'][f'prompt{i}']['do_generate_response'] = False

        # if st_state["prompts"][f"prompt{i}"]["RUN"] or st_state.generate_all:
        # st_state['prompts'][f'prompt{i}']['response'] = rep    # Update the response for index i
        # st.session_state['prompts']                   = st_state['prompts']  # Update the session state with the updated prompts
        # if st_state['prompts'][f'prompt{i}']['full'] is not None:
        # resp = st_state['prompts'][f'prompt{i}']['full'].choices[0]["msg_content"]["content"]
        #         st.write(resp)

        # Usage
        for i in range(st_state["row_count"]):

            init_prompt(st_state, i)
            display_inputs(st_state, i)
            st.divider()
        st_state["generate_all"] = False


def init_prompt(st_state, i):
    if i is None:
        return
    st_state[f"prompt{i}"] = " "
    if st_state[f"prompt{i}"] is None:
        st_state["prompts"][f"prompt{i}"] = {}
    notes                                                = st_state["prompts"][f"prompt{i}"]["notes"]
    msg_content                                          = st_state["prompts"][f"prompt{i}"]["msg_content"]
    response                                             = st_state["prompts"][f"prompt{i}"]["response"]
    prompt_tokens                                        = st_state["prompts"][f"prompt{i}"]["prompt_tokens"]
    response_tokens                                      = st_state["prompts"][f"prompt{i}"]["response_tokens"]
    word_count                                           = st_state["prompts"][f"prompt{i}"]["word_count"]
    response                                             = st_state[f"resp{i}"]
    st_state["prompts"][f"prompt{i}"]["notes"]           = notes or ""
    st_state["prompts"][f"prompt{i}"]["msg_content"]     = msg_content or ""
    st_state["prompts"][f"prompt{i}"]["response"]        = response or ""
    st_state["prompts"][f"prompt{i}"]["prompt_tokens"]   = prompt_tokens or 0
    st_state["prompts"][f"prompt{i}"]["response_tokens"] = response_tokens or 0
    st_state["prompts"][f"prompt{i}"]["word_count"]      = word_count or 0
    st_state[f"resp{i}"]                                 = response or "Response "


def create_download_link(text, filename):
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'


def show_dl(st_state):
    row_count      = st_state.row_count
    responses_text = "\n\n".join(
        [
            f"{ st_state['prompts'][f'prompt{i}']['notes'] or ''}\n{st_state['prompts'][f'prompt{i}']['response'] or ''}"
            for i in range(row_count)
            if show_notes
        ]
        + [
            st_state["prompts"][f"prompt{i}"]["response"]
            for i in range(row_count)
            if not show_notes
        ]
    )
    download_filename = "responses.txt"
    st.sidebar.download_button("Download ALL", responses_text)


# set these to init once then not again until refresh/restart


st_state['Initialized'] =  'Initialized' in st_state or False

if not st_state['Initialized']:
    Initializer(st_state)
    st_state['Initialized'] = True

openai.api_key   = get_api_key(st_state)

# cache the state
st.cache(lambda: st_state, allow_output_mutation=True)

main(st_state)
