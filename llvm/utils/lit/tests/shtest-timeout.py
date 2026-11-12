# REQUIRES: lit-max-individual-test-time

# llvm.org/PR33944
# UNSUPPORTED: system-windows

###############################################################################
# Check tests can hit timeout when set
###############################################################################

# Test per test timeout
# RUN: not %{lit} \
# RUN: %{inputs}/shtest-timeout/infinite_loop.py \
# RUN: -j 1 -v --debug --timeout 1 > %t.intsh.out
# RUN: FileCheck  --check-prefix=CHECK-OUT-COMMON < %t.intsh.out %s
# RUN: FileCheck --check-prefix=CHECK-INTSH-OUT < %t.intsh.out %s

# CHECK-INTSH-OUT: TIMEOUT: per_test_timeout :: infinite_loop.py
# CHECK-INTSH-OUT: command reached timeout: True

# Test per test timeout set via a config file rather than on the command line
# RUN: not %{lit} \
# RUN: %{inputs}/shtest-timeout/infinite_loop.py \
# RUN: -j 1 -v --debug \
# RUN: --param set_timeout=1 > %t.cfgset.out
# RUN: FileCheck --check-prefix=CHECK-OUT-COMMON  < %t.cfgset.out %s

# CHECK-OUT-COMMON: TIMEOUT: per_test_timeout :: infinite_loop.py
# CHECK-OUT-COMMON: Timeout: Reached timeout of 1 seconds
# CHECK-OUT-COMMON: Timed Out: 1


###############################################################################
# Check tests can complete in with a timeout set
#
# `short.py` should execute quickly so we shouldn't wait anywhere near the
# 3600 second timeout.
###############################################################################

# Test per test timeout
# RUN: %{lit} \
# RUN: %{inputs}/shtest-timeout/short.py \
# RUN: -j 1 -v --debug --timeout 3600 > %t.pass.intsh.out
# RUN: FileCheck  --check-prefix=CHECK-OUT-COMMON-SHORT < %t.pass.intsh.out %s

# CHECK-OUT-COMMON-SHORT: PASS: per_test_timeout :: short.py
# CHECK-OUT-COMMON-SHORT: Passed: 1

# Test per test timeout via a config file and on the command line.
# The value set on the command line should override the config file.
# RUN: %{lit} \
# RUN:   %{inputs}/shtest-timeout/short.py \
# RUN:   -j 1 -v --debug \
# RUN: --param set_timeout=1 --timeout=3600 > %t.pass.cmdover.out 2> %t.pass.cmdover.err
# RUN: FileCheck --check-prefix=CHECK-OUT-COMMON-SHORT  < %t.pass.cmdover.out %s
# RUN: FileCheck --check-prefix=CHECK-CMDLINE-OVERRIDE-ERR < %t.pass.cmdover.err %s

# CHECK-CMDLINE-OVERRIDE-ERR: Forcing timeout to be 3600 seconds
