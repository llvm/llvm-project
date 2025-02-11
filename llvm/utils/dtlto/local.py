import subprocess
import sys
import json
from pathlib import Path

if __name__ == "__main__":
    # Load the DTLTO information from the input JSON file.
    data = json.loads(Path(sys.argv[-1]).read_bytes())

    # Iterate over the jobs and execute the codegen tool.
    for job in data["jobs"]:
        jobargs = []
        for arg in data["common"]["args"]:
            if isinstance(arg, list):
                # arg is a "template", into which an external filename is to be
                # inserted. The first element of arg names an array of strings
                # in the job. The remaining elements of arg are either indices
                # into the array or literal strings.
                files, rest = job[arg[0]], arg[1:]
                jobargs.append(
                    "".join(files[x] if isinstance(x, int) else x for x in rest)
                )
            else:
                jobargs.append(arg)
        subprocess.check_call(jobargs)
