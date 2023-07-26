# Check that the environment variable is set correctly
# RUN: %{python} %s | FileCheck %s

# Python script to read the environment variable
# and print its value
import os

llvm_profile_file = os.environ.get('LLVM_PROFILE_FILE')
print(llvm_profile_file)

# CHECK: per-test-coverage0.profraw
