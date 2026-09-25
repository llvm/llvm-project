# Test LLVM_PROFILE_FILE is set when --per-test-coverage is passed to command line.

# RUN: %{lit} -a --per-test-coverage \
# RUN:     %{inputs}/per-test-coverage/per-test-coverage.py | \
# RUN:   FileCheck -DOUT=stdout %s

#      CHECK: {{^}}PASS: per-test-coverage :: per-test-coverage.py ({{[^)]*}})
#      CHECK: Command Output ([[OUT]]):
# CHECK-NEXT: --
#      CHECK: export
#      CHECK: LLVM_PROFILE_FILE=per-test-coverage.py-%p-%m0.profraw
#      CHECK: per-test-coverage.py
#      CHECK: {{RUN}}: at line 2
#      CHECK: export
#      CHECK: LLVM_PROFILE_FILE=per-test-coverage.py-%p-%m1.profraw
#      CHECK: per-test-coverage.py
#      CHECK: {{RUN}}: at line 3
#      CHECK: export
#      CHECK: LLVM_PROFILE_FILE=per-test-coverage.py-%p-%m2.profraw
#      CHECK: per-test-coverage.py
