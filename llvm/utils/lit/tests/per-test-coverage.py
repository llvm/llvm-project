# Test LLVM_PROFILE_FILE is set when --per-test-coverage is passed to command line.

# RUN: %{lit} -a -v --per-test-coverage %{inputs}/per-test-coverage/per-test-coverage.py \
# RUN: | FileCheck -match-full-lines %s
#
# CHECK: PASS: per-test-coverage :: per-test-coverage.py ({{[^)]*}})
