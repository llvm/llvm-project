# Check we don't add a GoogleTest binary more than once and issue a warning
# when it happens.

# RUN: %{lit} -v --order=random %{inputs}/googletest-detect-duplicate \
# RUN:                          %{inputs}/googletest-detect-duplicate 2>&1 | FileCheck %s

# CHECK: warning: Skip adding
# CHECK: -- Testing:
# CHECK: PASS: googletest-detect-duplicate :: [[PATH:[Dd]ummy[Ss]ub[Dd]ir/]][[FILE:OneTest\.py]]/0
# CHECK: Passed{{ *}}: 1
