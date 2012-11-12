"""
Defines utilities useful for performing standard "configuration" style tasks.
"""

import re
import os

def configure_file(input_path, output_path, substitutions):
    """configure_file(input_path, output_path, substitutions) -> bool

    Given an input and output path, "configure" the file at the given input path
    by replacing variables in the file with those given in the substitutions
    list. Returns true if the output file was written.

    The substitutions list should be given as a list of tuples (regex string,
    replacement), where the regex and replacement will be used as in 're.sub' to
    execute the variable replacement.

    The output path's parent directory need not exist (it will be created).

    If the output path does exist and the configured data is not different than
    it's current contents, the output file will not be modified. This is
    designed to limit the impact of configured files on build dependencies.
    """

    # Read in the input data.
    f = open(input_path, "rb")
    try:
        data = f.read()
    finally:
        f.close()

    # Perform the substitutions.
    for regex_string,replacement in substitutions:
        regex = re.compile(regex_string)
        data = regex.sub(replacement, data)

    # Ensure the output parent directory exists.
    output_parent_path = os.path.dirname(os.path.abspath(output_path))
    if not os.path.exists(output_parent_path):
        os.makedirs(output_parent_path)

    # If the output path exists, load it and compare to the configured contents.
    if os.path.exists(output_path):
        current_data = None
        try:
            f = open(output_path, "rb")
            try:
                current_data = f.read()
            except:
                current_data = None
            f.close()
        except:
            current_data = None

        if current_data is not None and current_data == data:
            return False

    # Write the output contents.
    f = open(output_path, "wb")
    try:
        f.write(data)
    finally:
        f.close()

    return True
