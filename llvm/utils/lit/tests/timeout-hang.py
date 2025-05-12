# REQUIRES: lit-max-individual-test-time

# Python has some issues dealing with exceptions when multiprocessing,
# which can cause hangs. Previously this could occur when we encountered
# an internal shell exception, and had a timeout set.

# This test runs a lit test that tries to launch a non-existent file,
# throwing an exception. We expect this to fail immediately, rather than
# timeout.

# DEFINE: %{timeout}=1

# RUN: not %{lit} %{inputs}/timeout-hang/run-nonexistent.txt \
# RUN: --timeout=%{timeout} --param external=0 | %{python} %s %{timeout}

import sys
import re

timeout_time = float(sys.argv[1])
testing_time = float(re.search(r"Testing Time: (.*)s", sys.stdin.read()).group(1))

if testing_time < timeout_time:
    print("Testing took less than timeout")
    sys.exit(0)
else:
    print("Testing took as long or longer than timeout")
    sys.exit(1)
