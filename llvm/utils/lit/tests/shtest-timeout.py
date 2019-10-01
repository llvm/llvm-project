# REQUIRES: lit-max-individual-test-time

# llvm.org/PR33944
# UNSUPPORTED: system-windows

# FIXME: This test is fragile because it relies on time which can
# be affected by system performance. In particular we are currently
# assuming that `short.py` can be successfully executed within 2
# seconds of wallclock time.

# Test per test timeout using external shell
# RUN: not %{lit} \
# RUN: %{inputs}/shtest-timeout/infinite_loop.py \
# RUN: %{inputs}/shtest-timeout/short.py \
# RUN: -j 1 -v --debug --timeout 2 --param external=1 > %t.extsh.out 2> %t.extsh.err
# RUN: FileCheck --check-prefix=CHECK-OUT-COMMON < %t.extsh.out %s
# RUN: FileCheck --check-prefix=CHECK-EXTSH-ERR < %t.extsh.err %s
#
# CHECK-EXTSH-ERR: Using external shell

# Test per test timeout using internal shell
# RUN: not %{lit} \
# RUN: %{inputs}/shtest-timeout/infinite_loop.py \
# RUN: %{inputs}/shtest-timeout/short.py \
# RUN: -j 1 -v --debug --timeout 2 --param external=0 > %t.intsh.out 2> %t.intsh.err
# RUN: FileCheck  --check-prefix=CHECK-OUT-COMMON < %t.intsh.out %s
# RUN: FileCheck --check-prefix=CHECK-INTSH-OUT < %t.intsh.out %s
# RUN: FileCheck --check-prefix=CHECK-INTSH-ERR < %t.intsh.err %s

# CHECK-INTSH-OUT: TIMEOUT: per_test_timeout :: infinite_loop.py
# CHECK-INTSH-OUT: command output:
# CHECK-INTSH-OUT: command reached timeout: True

# CHECK-INTSH-ERR: Using internal shell

# Test per test timeout set via a config file rather than on the command line
# RUN: not %{lit} \
# RUN: %{inputs}/shtest-timeout/infinite_loop.py \
# RUN: %{inputs}/shtest-timeout/short.py \
# RUN: -j 1 -v --debug --param external=0 \
# RUN: --param set_timeout=2 > %t.cfgset.out 2> %t.cfgset.err
# RUN: FileCheck --check-prefix=CHECK-OUT-COMMON  < %t.cfgset.out %s
# RUN: FileCheck --check-prefix=CHECK-CFGSET-ERR < %t.cfgset.err %s
#
# CHECK-CFGSET-ERR: Using internal shell

# CHECK-OUT-COMMON: TIMEOUT: per_test_timeout :: infinite_loop.py
# CHECK-OUT-COMMON: Timeout: Reached timeout of 2 seconds
# CHECK-OUT-COMMON: Command {{([0-9]+ )?}}Output

# CHECK-OUT-COMMON: PASS: per_test_timeout :: short.py

# CHECK-OUT-COMMON: Expected Passes{{ *}}: 1
# CHECK-OUT-COMMON: Individual Timeouts{{ *}}: 1

# Test per test timeout via a config file and on the command line.
# The value set on the command line should override the config file.
# RUN: not %{lit} \
# RUN: %{inputs}/shtest-timeout/infinite_loop.py \
# RUN: %{inputs}/shtest-timeout/short.py \
# RUN: -j 1 -v --debug --param external=0 \
# RUN: --param set_timeout=1 --timeout=2 > %t.cmdover.out 2> %t.cmdover.err
# RUN: FileCheck --check-prefix=CHECK-CMDLINE-OVERRIDE-OUT  < %t.cmdover.out %s
# RUN: FileCheck --check-prefix=CHECK-CMDLINE-OVERRIDE-ERR < %t.cmdover.err %s

# CHECK-CMDLINE-OVERRIDE-ERR: Forcing timeout to be 2 seconds

# CHECK-CMDLINE-OVERRIDE-OUT: TIMEOUT: per_test_timeout :: infinite_loop.py
# CHECK-CMDLINE-OVERRIDE-OUT: Timeout: Reached timeout of 2 seconds
# CHECK-CMDLINE-OVERRIDE-OUT: Command {{([0-9]+ )?}}Output

# CHECK-CMDLINE-OVERRIDE-OUT: PASS: per_test_timeout :: short.py

# CHECK-CMDLINE-OVERRIDE-OUT: Expected Passes{{ *}}: 1
# CHECK-CMDLINE-OVERRIDE-OUT: Individual Timeouts{{ *}}: 1
