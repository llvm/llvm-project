# Test if lit_config.per_test_coverage in lit.cfg sets individual test case coverage.

# RUN: %{lit} -a %{inputs}/per-test-coverage-by-lit-cfg/per-test-coverage-by-lit-cfg.py | \
# RUN:   FileCheck -DOUT=stdout %s

#      CHECK: {{^}}PASS: per-test-coverage-by-lit-cfg :: per-test-coverage-by-lit-cfg.py ({{[^)]*}})
#      CHECK: Command Output ([[OUT]]):
# CHECK-NEXT: --
#      CHECK: export
#      CHECK: LLVM_PROFILE_FILE=per-test-coverage-by-lit-cfg.py-%p-%m0.profraw
#      CHECK: per-test-coverage-by-lit-cfg.py
#      CHECK: {{RUN}}: at line 2
#      CHECK: export
#      CHECK: LLVM_PROFILE_FILE=per-test-coverage-by-lit-cfg.py-%p-%m1.profraw
#      CHECK: per-test-coverage-by-lit-cfg.py
#      CHECK: {{RUN}}: at line 3
#      CHECK: export
#      CHECK: LLVM_PROFILE_FILE=per-test-coverage-by-lit-cfg.py-%p-%m2.profraw
#      CHECK: per-test-coverage-by-lit-cfg.py

# Sibling tests sharing a basename in different directories must get distinct profile filenames.
# RUN: %{lit} -a -Dexecute_external=False \
# RUN:     %{inputs}/per-test-coverage-by-lit-cfg/name-collision | \
# RUN:   FileCheck -check-prefix=COLLISION %s

# COLLISION-DAG: LLVM_PROFILE_FILE=name-collision_a_test.py-%p-%m0.profraw
# COLLISION-DAG: LLVM_PROFILE_FILE=name-collision_b_test.py-%p-%m0.profraw
