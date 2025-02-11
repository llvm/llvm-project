import sys
import json
import shutil
from pathlib import Path

if __name__ == "__main__":
    json_arg = sys.argv[-1]
    distributor_args = sys.argv[1:-1]

    # Load the DTLTO information from the input JSON file.
    data = json.loads(Path(json_arg).read_bytes())

    # Iterate over the jobs and create the output
    # files by copying over the supplied input files.
    for job_index, job in enumerate(data["jobs"]):
        shutil.copy(distributor_args[job_index], job["primary_output"][0])
