# REQUIRES: lit-max-individual-test-time

# Python has some issues dealing with exceptions when multiprocessing,
# which can cause hangs. Previously this could occur when we encountered
# an internal shell exception, and had a timeout set.

# This test runs a lit test that tries to launch a non-existent file,
# throwing an exception. We expect this to fail immediately, rather than
# timeout.

# lit should return immediately once it fails to execute the non-existent file.
# This will take a variable amount of time depending on process scheduling, but
# it should always be significantly less than the hard timeout, which is the
# point where lit would cancel the test.
# DEFINE: %{grace_period}=5
# DEFINE: %{hard_timeout}=15

# RUN: not %{lit} %{inputs}/timeout-hang/run-nonexistent.txt \
# RUN: --timeout=%{hard_timeout} --param external=0 | %{python} %s %{grace_period}

import sys
import re

grace_time = float(sys.argv[1])
testing_time = float(re.search(r"Testing Time: (.*)s", sys.stdin.read()).group(1))

if testing_time <= grace_time:
    print("Testing finished within the grace period")
    sys.exit(0)
else:
    print(
        "Testing took {}s, which is beyond the grace period of {}s".format(
            testing_time, grace_time
        )
    )
    sys.exit(1)
