# RUN: %{python} %s | FileCheck -DINDEX=1 %s
# RUN: %{python} %s | FileCheck -DINDEX=2 %s

import os

print(os.environ.get("LLVM_PROFILE_FILE"))

# CHECK: name-collision_b_test.py-%p-%m[[INDEX]].profraw
