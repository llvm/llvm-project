# Check the behavior of the ALLOW_RETRIES keyword.

# This test uses a file that's stable across retries of the test to fail and
# only succeed the fourth time it is retried.
#
# RUN: rm -f %t.counter
# RUN: %{lit} %{inputs}/allow-retries/succeeds-within-limit.py -Dcounter=%t.counter -Dpython=%{python} | FileCheck --check-prefix=CHECK-TEST1 %s
# CHECK-TEST1: Passed With Retry: 1

# Test that a per-file ALLOW_RETRIES overwrites the config-wide test_retry_attempts property, if any.
#
# RUN: rm -f %t.counter
# RUN: %{lit} %{inputs}/allow-retries/succeeds-within-limit.py -Dtest_retry_attempts=2 -Dcounter=%t.counter -Dpython=%{python} | FileCheck --check-prefix=CHECK-TEST2 %s
# CHECK-TEST2: Passed With Retry: 1

# This test does not succeed within the allowed retry limit
#
# RUN: not %{lit} %{inputs}/allow-retries/does-not-succeed-within-limit.py | FileCheck --check-prefix=CHECK-TEST3 %s
# CHECK-TEST3: Failed Tests (1):
# CHECK-TEST3: allow-retries :: does-not-succeed-within-limit.py

# This test should be UNRESOLVED since it has more than one ALLOW_RETRIES
# lines, and that is not allowed.
#
# RUN: not %{lit} %{inputs}/allow-retries/more-than-one-allow-retries-lines.py | FileCheck --check-prefix=CHECK-TEST4 %s
# CHECK-TEST4: Unresolved Tests (1):
# CHECK-TEST4: allow-retries :: more-than-one-allow-retries-lines.py

# This test does not provide a valid integer to the ALLOW_RETRIES keyword.
# It should be unresolved.
#
# RUN: not %{lit} %{inputs}/allow-retries/not-a-valid-integer.py | FileCheck --check-prefix=CHECK-TEST5 %s
# CHECK-TEST5: Unresolved Tests (1):
# CHECK-TEST5: allow-retries :: not-a-valid-integer.py

# This test checks that the config-wide test_retry_attempts property is used
# when no ALLOW_RETRIES keyword is present.
#
# RUN: rm -f %t.counter
# RUN: %{lit} %{inputs}/test_retry_attempts/test.py -Dcounter=%t.counter -Dpython=%{python} | FileCheck --check-prefix=CHECK-TEST6 %s
# CHECK-TEST6: Passed With Retry: 1
