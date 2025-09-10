import os
import subprocess
import sys

os.environ["SET_IN_PARENT"] = "something"
out = subprocess.run([sys.argv[1], sys.argv[2]])
sys.exit(out.returncode)
