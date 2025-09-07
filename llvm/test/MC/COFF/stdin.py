# RUN: echo "// comment" > %t.input
# RUN: which llvm-mc | %python %s %t

import subprocess
import sys

llvm_mc_binary = sys.stdin.readlines()[0].strip()
temp_file = sys.argv[1]
input_file = temp_file + ".input"

with open(temp_file, "w") as mc_stdout:
    mc_stdout.seek(4)
    subprocess.run(
        [llvm_mc_binary, "-filetype=obj", "-triple", "i686-pc-win32", input_file],
        stdout=mc_stdout,
        check=True,
    )
