# Test if lit_config.per_test_coverage in lit.cfg sets individual test case coverage.

# RUN: %{lit} -a -v %{inputs}/per-test-coverage-by-lit-cfg/per-test-coverage-by-lit-cfg.py \
# RUN: | FileCheck -match-full-lines %s
#
# CHECK: PASS: per-test-coverage-by-lit-cfg :: per-test-coverage-by-lit-cfg.py ({{[^)]*}})
