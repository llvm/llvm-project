# REQUIRES: lit-max-individual-test-time
# UNSUPPORTED: system-windows

# RUN: not %{lit} \
# RUN:   %{inputs}/timeout-config \
# RUN:   -j 1 -v > %t.out 2> %t.err
# RUN: FileCheck < %t.out %s

# CHECK: TIMEOUT: timeout-config :: test.py
