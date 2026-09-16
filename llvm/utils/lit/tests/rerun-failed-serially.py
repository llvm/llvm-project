# Check the behavior of --rerun-failed-serially.

# The test suite used here contains a test that fails the first time it is run
# and passes on the rerun, and one that fails every time.

# Without the option, the test that would pass on a rerun is just a failure.
#
# RUN: rm -f %t.counter
# RUN: not %{lit} %{inputs}/rerun-failed-serially -Dcounter=%t.counter \
# RUN:     -Dpython=%{python} 2>&1 | \
# RUN:   FileCheck --check-prefix=CHECK-OFF %s
#
#      CHECK-OFF: Failed Tests (2):
# CHECK-OFF-NEXT: rerun-failed-serially :: always-fails.py
# CHECK-OFF-NEXT: rerun-failed-serially :: fails-once.py
#      CHECK-OFF: Failed: 2

# With a pattern that matches every failure, the failed tests are run again
# with a single worker. The one that passes the second time is reported as
# flaky, the other one still fails.
#
# RUN: rm -f %t.counter
# RUN: not %{lit} %{inputs}/rerun-failed-serially --rerun-failed-serially . \
# RUN:     -Dcounter=%t.counter -Dpython=%{python} 2>&1 | \
# RUN:   FileCheck --check-prefix=CHECK-ON %s
#
#      CHECK-ON: note: rerunning 2 failed test(s) with a single worker
#      CHECK-ON: -- Testing: 2 tests, 1 workers --
#      CHECK-ON: Failed Tests (1):
# CHECK-ON-NEXT: rerun-failed-serially :: always-fails.py
#      CHECK-ON: Passed With Retry: 1
#      CHECK-ON: Failed{{ *}}: 1

# A narrower pattern restricts the rerun to the failures whose output matches,
# so that a test failing for an unrelated reason is not given a second chance.
#
# RUN: rm -f %t.counter
# RUN: not %{lit} %{inputs}/rerun-failed-serially \
# RUN:     --rerun-failed-serially "machine was busy" \
# RUN:     -Dcounter=%t.counter -Dpython=%{python} 2>&1 | \
# RUN:   FileCheck --check-prefix=CHECK-MATCHING %s
#
#      CHECK-MATCHING: note: rerunning 1 failed test(s) with a single worker
#      CHECK-MATCHING: Failed Tests (1):
# CHECK-MATCHING-NEXT: rerun-failed-serially :: always-fails.py
#      CHECK-MATCHING: Passed With Retry: 1
#      CHECK-MATCHING: Failed{{ *}}: 1

# A pattern that matches no failure at all reruns nothing.
#
# RUN: rm -f %t.counter
# RUN: not %{lit} %{inputs}/rerun-failed-serially \
# RUN:     --rerun-failed-serially "no failure says this" \
# RUN:     -Dcounter=%t.counter -Dpython=%{python} 2>&1 | \
# RUN:   FileCheck --check-prefix=CHECK-NO-MATCH %s
#
#      CHECK-NO-MATCH-NOT: rerunning
#          CHECK-NO-MATCH: Failed Tests (2):
#     CHECK-NO-MATCH-NEXT: rerun-failed-serially :: always-fails.py
#     CHECK-NO-MATCH-NEXT: rerun-failed-serially :: fails-once.py
#          CHECK-NO-MATCH: Failed: 2

# The in-place retries of --max-retries-per-test come first, so only a failure
# that outlives them reaches the serial rerun. The test used below needs four
# attempts to pass; with two attempts per pass, the rerun is what gets it there
# and lit reports the flaky pass itself.
#
# RUN: rm -f %t.counter
# RUN: %{lit} %{inputs}/rerun-failed-serially-retries --max-retries-per-test 1 \
# RUN:     --rerun-failed-serially "machine was busy" \
# RUN:     -Dcounter=%t.counter -Dpython=%{python} 2>&1 | \
# RUN:   FileCheck --check-prefix=CHECK-RETRIES %s
#
# CHECK-RETRIES: FAIL: rerun-failed-serially-retries :: needs-four-attempts.py (1 of 1, 2 of 2 attempts)
# CHECK-RETRIES: note: rerunning 1 failed test(s) with a single worker
# CHECK-RETRIES: FLAKYPASS: rerun-failed-serially-retries :: needs-four-attempts.py (1 of 1, 2 of 2 attempts)
# CHECK-RETRIES: Passed With Retry: 1

# With three attempts per pass, the fourth one is the first attempt of the
# rerun, which passes, so the flaky pass comes from the rerun instead.
#
# RUN: rm -f %t.counter
# RUN: %{lit} %{inputs}/rerun-failed-serially-retries --max-retries-per-test 2 \
# RUN:     --rerun-failed-serially "machine was busy" \
# RUN:     -Dcounter=%t.counter -Dpython=%{python} 2>&1 | \
# RUN:   FileCheck --check-prefix=CHECK-RETRIES-PASS %s
#
# CHECK-RETRIES-PASS: FAIL: rerun-failed-serially-retries :: needs-four-attempts.py (1 of 1, 3 of 3 attempts)
# CHECK-RETRIES-PASS: note: rerunning 1 failed test(s) with a single worker
# CHECK-RETRIES-PASS: PASS: rerun-failed-serially-retries :: needs-four-attempts.py (1 of 1)
# CHECK-RETRIES-PASS: Passed With Retry: 1
