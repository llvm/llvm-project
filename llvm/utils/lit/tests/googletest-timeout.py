# REQUIRES: lit-max-individual-test-time

###############################################################################
# Check tests can hit timeout when set
###############################################################################

# Check that the per test timeout is enforced when running GTest tests.
#
# RUN: not %{lit} -j 1 -v %{inputs}/googletest-timeout \
# RUN:   --filter=InfiniteLoopSubTest --timeout=1 > %t.cmd.out
# RUN: FileCheck --check-prefix=CHECK-INF < %t.cmd.out %s

# Check that the per test timeout is enforced when running GTest tests via
# the configuration file
#
# RUN: not %{lit} -j 1 -v %{inputs}/googletest-timeout \
# RUN:  --filter=InfiniteLoopSubTest  --param set_timeout=1 \
# RUN:  > %t.cfgset.out
# RUN: FileCheck --check-prefix=CHECK-INF < %t.cfgset.out %s

# CHECK-INF: -- Testing:
# CHECK-INF: TIMEOUT: googletest-timeout :: [[PATH:[Dd]ummy[Ss]ub[Dd]ir/]][[FILE:OneTest\.py]]/[[TEST:T\.InfiniteLoopSubTest]]
# CHECK-INF-NEXT: ******************** TEST 'googletest-timeout :: [[PATH]][[FILE]]/[[TEST]]' FAILED ********************
# CHECK-INF-NEXT: Script:
# CHECK-INF-NEXT: --
# CHECK-INF-NEXT: [[FILE]] --gtest_filter=[[TEST]]
# CHECK-INF-NEXT: --
# CHECK-INF: Timed Out: 1

###############################################################################
# Check tests can complete with a timeout set
#
# `QuickSubTest` should execute quickly so we shouldn't wait anywhere near the
# 3600 second timeout.
###############################################################################

# RUN: %{lit} -j 1 -v %{inputs}/googletest-timeout \
# RUN:   --filter=QuickSubTest --timeout=3600 > %t.cmd.out
# RUN: FileCheck --check-prefix=CHECK-QUICK < %t.cmd.out %s

# CHECK-QUICK: PASS: googletest-timeout :: {{[Dd]ummy[Ss]ub[Dd]ir}}/OneTest.py/T.QuickSubTest
# CHECK-QUICK: Passed : 1

# Test per test timeout via a config file and on the command line.
# The value set on the command line should override the config file.
# RUN: %{lit} -j 1 -v %{inputs}/googletest-timeout --filter=QuickSubTest \
# RUN:   --param set_timeout=1 --timeout=3600 \
# RUN:   > %t.cmdover.out 2> %t.cmdover.err
# RUN: FileCheck --check-prefix=CHECK-QUICK < %t.cmdover.out %s
# RUN: FileCheck --check-prefix=CHECK-CMDLINE-OVERRIDE-ERR < %t.cmdover.err %s

# CHECK-CMDLINE-OVERRIDE-ERR: Forcing timeout to be 3600 seconds
