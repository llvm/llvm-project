# Check that --pass-env passes the named environment variable through to the
# test environment.
#
# RUN: env LIT_TEST_VAR=passed_value \
# RUN:   %{lit} -a %{inputs}/pass-env --pass-env=LIT_TEST_VAR \
# RUN:   | FileCheck -check-prefix=PASSED %s

# Check that --pass-env can be repeated and also works via LIT_OPTS.
#
# RUN: env LIT_TEST_VAR=passed_value LIT_TEST_VAR2=passed_value2 LIT_OPTS=--pass-env=LIT_TEST_VAR2 \
# RUN:   %{lit} -a %{inputs}/pass-env --pass-env=LIT_TEST_VAR \
# RUN:   | FileCheck -check-prefix=BOTH %s

# Check that without --pass-env, a non-allow-listed variable is not passed
# through to the test environment.
#
# RUN: env LIT_TEST_VAR=passed_value \
# RUN:   %{lit} -a %{inputs}/pass-env \
# RUN:   | FileCheck -check-prefix=NOTPASSED %s

# PASSED: LIT_TEST_VAR=passed_value

# BOTH-DAG: LIT_TEST_VAR=passed_value
# BOTH-DAG: LIT_TEST_VAR2=passed_value2

# NOTPASSED-NOT: LIT_TEST_VAR=passed_value
