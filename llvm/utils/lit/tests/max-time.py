# Test overall lit timeout (--max-time).
#
# RUN: %{lit} %{inputs}/max-time --max-time=1 2>&1  |  FileCheck %s

# CHECK: reached timeout, skipping remaining tests
# CHECK: Skipped Tests  : 1
# CHECK: Expected Passes: 1
