# Test various combinations of options controlling lit stdout and stderr output

# RUN: mkdir -p %t

### Test default

# RUN: not %{lit} %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# NO-ARGS:      -- Testing: 5 tests, 1 workers --
# NO-ARGS-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# NO-ARGS-NEXT: PASS: verbosity :: pass.txt (2 of 5)
# NO-ARGS-NEXT: {{UN}}SUPPORTED: verbosity :: unsupported.txt (3 of 5)
# NO-ARGS-NEXT: {{X}}FAIL: verbosity :: xfail.txt (4 of 5)
# NO-ARGS-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# NO-ARGS-NEXT: ********************
# NO-ARGS-NEXT: Failed Tests (1):
# NO-ARGS-NEXT:   verbosity :: fail.txt
# NO-ARGS-EMPTY:
# NO-ARGS-NEXT: ********************
# NO-ARGS-NEXT: Unexpectedly Passed Tests (1):
# NO-ARGS-NEXT:   verbosity :: xpass.txt
# NO-ARGS-EMPTY:
# NO-ARGS-EMPTY:
# NO-ARGS-NEXT: Testing Time: {{.*}}s
# NO-ARGS-EMPTY:
# NO-ARGS-NEXT: Total Discovered Tests: 5
# NO-ARGS-NEXT:   Unsupported        : 1 (20.00%)
# NO-ARGS-NEXT:   Passed             : 1 (20.00%)
# NO-ARGS-NEXT:   Expectedly Failed  : 1 (20.00%)
# NO-ARGS-NEXT:   Failed             : 1 (20.00%)
# NO-ARGS-NEXT:   Unexpectedly Passed: 1 (20.00%)

# NO-ARGS-ERR: lit.py: {{.*}}lit.cfg:{{[0-9]+}}: note: this is a note
# NO-ARGS-ERR-NEXT: lit.py: {{.*}}lit.cfg:{{[0-9]+}}: warning: this is a warning
# NO-ARGS-ERR-EMPTY:
# NO-ARGS-ERR-NEXT: 1 warning(s) in tests


### Test aliases

# RUN: not %{lit} --succinct %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix SUCCINCT < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# SUCCINCT:      -- Testing: 5 tests, 1 workers --
# SUCCINCT-NEXT: Testing:
# SUCCINCT-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# SUCCINCT-NEXT: Testing:
# SUCCINCT-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# SUCCINCT-NEXT: Testing:
# SUCCINCT-NEXT: ********************
# SUCCINCT-NEXT: Failed Tests (1):
# SUCCINCT-NEXT:   verbosity :: fail.txt
# SUCCINCT-EMPTY:
# SUCCINCT-NEXT: ********************
# SUCCINCT-NEXT: Unexpectedly Passed Tests (1):
# SUCCINCT-NEXT:   verbosity :: xpass.txt
# SUCCINCT-EMPTY:
# SUCCINCT-EMPTY:
# SUCCINCT-NEXT: Testing Time: {{.*}}s
# SUCCINCT-EMPTY:
# SUCCINCT-NEXT: Total Discovered Tests: 5
# SUCCINCT-NEXT:   Unsupported        : 1 (20.00%)
# SUCCINCT-NEXT:   Passed             : 1 (20.00%)
# SUCCINCT-NEXT:   Expectedly Failed  : 1 (20.00%)
# SUCCINCT-NEXT:   Failed             : 1 (20.00%)
# SUCCINCT-NEXT:   Unexpectedly Passed: 1 (20.00%)

# RUN: not %{lit} --verbose %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix VERBOSE < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# VERBOSE:      -- Testing: 5 tests, 1 workers --
# VERBOSE-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# VERBOSE-NEXT: ******************** TEST 'verbosity :: fail.txt' FAILED ********************
# VERBOSE-NEXT: Exit Code: 127
# VERBOSE-EMPTY:
# VERBOSE-NEXT: Command Output (stdout):
# VERBOSE-NEXT: --
# VERBOSE-NEXT: # {{R}}UN: at line 1
# VERBOSE-NEXT: echo "fail test output"
# VERBOSE-NEXT: # executed command: echo 'fail test output'
# VERBOSE-NEXT: # .---command stdout------------
# VERBOSE-NEXT: # | fail test output
# VERBOSE-NEXT: # `-----------------------------
# VERBOSE-NEXT: # {{R}}UN: at line 2
# VERBOSE-NEXT: fail
# VERBOSE-NEXT: # executed command: fail
# VERBOSE-NEXT: # .---command stderr------------
# VERBOSE-NEXT: # | 'fail': command not found
# VERBOSE-NEXT: # `-----------------------------
# VERBOSE-NEXT: # error: command failed with exit status: 127
# VERBOSE-EMPTY:
# VERBOSE-NEXT: --
# VERBOSE-EMPTY:
# VERBOSE-NEXT: ********************
# VERBOSE-NEXT: PASS: verbosity :: pass.txt (2 of 5)
# VERBOSE-NEXT: {{UN}}SUPPORTED: verbosity :: unsupported.txt (3 of 5)
# VERBOSE-NEXT: {{X}}FAIL: verbosity :: xfail.txt (4 of 5)
# VERBOSE-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# VERBOSE-NEXT: ******************** TEST 'verbosity :: xpass.txt' FAILED ********************
# VERBOSE-NEXT: Exit Code: 0
# VERBOSE-EMPTY:
# VERBOSE-NEXT: Command Output (stdout):
# VERBOSE-NEXT: --
# VERBOSE-NEXT: # {{R}}UN: at line 2
# VERBOSE-NEXT: echo "xpass test output"
# VERBOSE-NEXT: # executed command: echo 'xpass test output'
# VERBOSE-NEXT: # .---command stdout------------
# VERBOSE-NEXT: # | xpass test output
# VERBOSE-NEXT: # `-----------------------------
# VERBOSE-EMPTY:
# VERBOSE-NEXT: --
# VERBOSE-EMPTY:
# VERBOSE-NEXT: ********************
# VERBOSE-NEXT: ********************
# VERBOSE-NEXT: Failed Tests (1):
# VERBOSE-NEXT:   verbosity :: fail.txt
# VERBOSE-EMPTY:
# VERBOSE-NEXT: ********************
# VERBOSE-NEXT: Unexpectedly Passed Tests (1):
# VERBOSE-NEXT:   verbosity :: xpass.txt
# VERBOSE-EMPTY:
# VERBOSE-EMPTY:
# VERBOSE-NEXT: Testing Time: {{.*}}s
# VERBOSE-EMPTY:
# VERBOSE-NEXT: Total Discovered Tests: 5
# VERBOSE-NEXT:   Unsupported        : 1 (20.00%)
# VERBOSE-NEXT:   Passed             : 1 (20.00%)
# VERBOSE-NEXT:   Expectedly Failed  : 1 (20.00%)
# VERBOSE-NEXT:   Failed             : 1 (20.00%)
# VERBOSE-NEXT:   Unexpectedly Passed: 1 (20.00%)

# RUN: not %{lit} --show-all %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix SHOW-ALL < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# SHOW-ALL:      -- Testing: 5 tests, 1 workers --
# SHOW-ALL-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# SHOW-ALL-NEXT: ******************** TEST 'verbosity :: fail.txt' FAILED ********************
# SHOW-ALL-NEXT: Exit Code: 127
# SHOW-ALL-EMPTY:
# SHOW-ALL-NEXT: Command Output (stdout):
# SHOW-ALL-NEXT: --
# SHOW-ALL-NEXT: # {{R}}UN: at line 1
# SHOW-ALL-NEXT: echo "fail test output"
# SHOW-ALL-NEXT: # executed command: echo 'fail test output'
# SHOW-ALL-NEXT: # .---command stdout------------
# SHOW-ALL-NEXT: # | fail test output
# SHOW-ALL-NEXT: # `-----------------------------
# SHOW-ALL-NEXT: # {{R}}UN: at line 2
# SHOW-ALL-NEXT: fail
# SHOW-ALL-NEXT: # executed command: fail
# SHOW-ALL-NEXT: # .---command stderr------------
# SHOW-ALL-NEXT: # | 'fail': command not found
# SHOW-ALL-NEXT: # `-----------------------------
# SHOW-ALL-NEXT: # error: command failed with exit status: 127
# SHOW-ALL-EMPTY:
# SHOW-ALL-NEXT: --
# SHOW-ALL-EMPTY:
# SHOW-ALL-NEXT: ********************
# SHOW-ALL-NEXT: PASS: verbosity :: pass.txt (2 of 5)
# SHOW-ALL-NEXT: Exit Code: 0
# SHOW-ALL-EMPTY:
# SHOW-ALL-NEXT: Command Output (stdout):
# SHOW-ALL-NEXT: --
# SHOW-ALL-NEXT: # {{R}}UN: at line 1
# SHOW-ALL-NEXT: echo "pass test output"
# SHOW-ALL-NEXT: # executed command: echo 'pass test output'
# SHOW-ALL-NEXT: # .---command stdout------------
# SHOW-ALL-NEXT: # | pass test output
# SHOW-ALL-NEXT: # `-----------------------------
# SHOW-ALL-EMPTY:
# SHOW-ALL-NEXT: --
# SHOW-ALL-EMPTY:
# SHOW-ALL-NEXT: ********************
# SHOW-ALL-NEXT: {{UN}}SUPPORTED: verbosity :: unsupported.txt (3 of 5)
# SHOW-ALL-NEXT: Test requires the following unavailable features: asdf
# SHOW-ALL-NEXT: ********************
# SHOW-ALL-NEXT: {{X}}FAIL: verbosity :: xfail.txt (4 of 5)
# SHOW-ALL-NEXT: Exit Code: 1
# SHOW-ALL-EMPTY:
# SHOW-ALL-NEXT: Command Output (stdout):
# SHOW-ALL-NEXT: --
# SHOW-ALL-NEXT: # {{R}}UN: at line 2
# SHOW-ALL-NEXT: not echo "xfail test output"
# SHOW-ALL-NEXT: # executed command: not echo 'xfail test output'
# SHOW-ALL-NEXT: # .---command stdout------------
# SHOW-ALL-NEXT: # | xfail test output
# SHOW-ALL-NEXT: # `-----------------------------
# SHOW-ALL-NEXT: # error: command failed with exit status: 1
# SHOW-ALL-EMPTY:
# SHOW-ALL-NEXT: --
# SHOW-ALL-EMPTY:
# SHOW-ALL-NEXT: ********************
# SHOW-ALL-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# SHOW-ALL-NEXT: ******************** TEST 'verbosity :: xpass.txt' FAILED ********************
# SHOW-ALL-NEXT: Exit Code: 0
# SHOW-ALL-EMPTY:
# SHOW-ALL-NEXT: Command Output (stdout):
# SHOW-ALL-NEXT: --
# SHOW-ALL-NEXT: # {{R}}UN: at line 2
# SHOW-ALL-NEXT: echo "xpass test output"
# SHOW-ALL-NEXT: # executed command: echo 'xpass test output'
# SHOW-ALL-NEXT: # .---command stdout------------
# SHOW-ALL-NEXT: # | xpass test output
# SHOW-ALL-NEXT: # `-----------------------------
# SHOW-ALL-EMPTY:
# SHOW-ALL-NEXT: --
# SHOW-ALL-EMPTY:
# SHOW-ALL-NEXT: ********************
# SHOW-ALL-NEXT: ********************
# SHOW-ALL-NEXT: Failed Tests (1):
# SHOW-ALL-NEXT:   verbosity :: fail.txt
# SHOW-ALL-EMPTY:
# SHOW-ALL-NEXT: ********************
# SHOW-ALL-NEXT: Unexpectedly Passed Tests (1):
# SHOW-ALL-NEXT:   verbosity :: xpass.txt
# SHOW-ALL-EMPTY:
# SHOW-ALL-EMPTY:
# SHOW-ALL-NEXT: Testing Time: {{.*}}s
# SHOW-ALL-EMPTY:
# SHOW-ALL-NEXT: Total Discovered Tests: 5
# SHOW-ALL-NEXT:   Unsupported        : 1 (20.00%)
# SHOW-ALL-NEXT:   Passed             : 1 (20.00%)
# SHOW-ALL-NEXT:   Expectedly Failed  : 1 (20.00%)
# SHOW-ALL-NEXT:   Failed             : 1 (20.00%)
# SHOW-ALL-NEXT:   Unexpectedly Passed: 1 (20.00%)

# RUN: not %{lit} --quiet %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix QUIET < %t/stdout.txt
# RUN: FileCheck %s --check-prefix QUIET-ERR --implicit-check-not lit < %t/stderr.txt

# QUIET:      -- Testing: 5 tests, 1 workers --
# QUIET-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# QUIET-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# QUIET-NEXT: ********************
# QUIET-NEXT: Failed Tests (1):
# QUIET-NEXT:   verbosity :: fail.txt
# QUIET-EMPTY:
# QUIET-NEXT: ********************
# QUIET-NEXT: Unexpectedly Passed Tests (1):
# QUIET-NEXT:   verbosity :: xpass.txt
# QUIET-EMPTY:
# QUIET-EMPTY:
# QUIET-NEXT: Total Discovered Tests: 5
# QUIET-NEXT:   Failed             : 1 (20.00%)
# QUIET-NEXT:   Unexpectedly Passed: 1 (20.00%)

# QUIET-ERR: 1 warning(s) in tests


### Test log output

# RUN: not %{lit} --debug %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix DEBUG < %t/stdout.txt
# RUN: FileCheck %s --check-prefix DEBUG-ERR --implicit-check-not lit < %t/stderr.txt

# DEBUG:      -- Testing: 5 tests, 1 workers --
# DEBUG-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# DEBUG-NEXT: PASS: verbosity :: pass.txt (2 of 5)
# DEBUG-NEXT: {{UN}}SUPPORTED: verbosity :: unsupported.txt (3 of 5)
# DEBUG-NEXT: {{X}}FAIL: verbosity :: xfail.txt (4 of 5)
# DEBUG-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# DEBUG-NEXT: ********************
# DEBUG-NEXT: Failed Tests (1):
# DEBUG-NEXT:   verbosity :: fail.txt
# DEBUG-EMPTY:
# DEBUG-NEXT: ********************
# DEBUG-NEXT: Unexpectedly Passed Tests (1):
# DEBUG-NEXT:   verbosity :: xpass.txt
# DEBUG-EMPTY:
# DEBUG-EMPTY:
# DEBUG-NEXT: Testing Time: {{.*}}s
# DEBUG-EMPTY:
# DEBUG-NEXT: Total Discovered Tests: 5
# DEBUG-NEXT:   Unsupported        : 1 (20.00%)
# DEBUG-NEXT:   Passed             : 1 (20.00%)
# DEBUG-NEXT:   Expectedly Failed  : 1 (20.00%)
# DEBUG-NEXT:   Failed             : 1 (20.00%)
# DEBUG-NEXT:   Unexpectedly Passed: 1 (20.00%)

# DEBUG-ERR:      lit.py: {{.*}}discovery.py:{{[0-9]+}}: debug: loading suite config '{{.*}}lit.cfg'
# DEBUG-ERR-NEXT: lit.py: {{.*}}lit.cfg:{{[0-9]+}}: debug: this is a debug log
# DEBUG-ERR-NEXT: lit.py: {{.*}}lit.cfg:{{[0-9]+}}: note: this is a note
# DEBUG-ERR-NEXT: lit.py: {{.*}}lit.cfg:{{[0-9]+}}: warning: this is a warning
# DEBUG-ERR-NEXT: lit.py: {{.*}}TestingConfig.py:{{[0-9]+}}: debug: ... loaded config '{{.*}}lit.cfg'
# DEBUG-ERR-NEXT: lit.py: {{.*}}discovery.py:{{[0-9]+}}: debug: resolved input '{{.*}}verbosity' to 'verbosity'::()
# DEBUG-ERR-EMPTY:
# DEBUG-ERR-NEXT: 1 warning(s) in tests


# RUN: not %{lit} --diagnostic-level note %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# RUN: not %{lit} --diagnostic-level warning %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS < %t/stdout.txt
# RUN: FileCheck %s --check-prefix WARNING-ERR --implicit-check-not lit < %t/stderr.txt

# WARNING-ERR: lit.py: {{.*}}lit.cfg:{{[0-9]+}}: warning: this is a warning
# WARNING-ERR-EMPTY:
# WARNING-ERR-NEXT: 1 warning(s) in tests

# RUN: not %{lit} --diagnostic-level error %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS < %t/stdout.txt
# RUN: FileCheck %s --check-prefix ERROR-ERR --implicit-check-not lit < %t/stderr.txt

# ERROR-ERR: 1 warning(s) in tests


### Test --test-output

# RUN: not %{lit} --test-output off  %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# RUN: not %{lit} --test-output failed  %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix VERBOSE < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# TEST-OUTPUT-OFF:      -- Testing: 5 tests, 1 workers --
# TEST-OUTPUT-OFF-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# TEST-OUTPUT-OFF-NEXT: PASS: verbosity :: pass.txt (2 of 5)
# TEST-OUTPUT-OFF-NEXT: {{UN}}SUPPORTED: verbosity :: unsupported.txt (3 of 5)
# TEST-OUTPUT-OFF-NEXT: {{X}}FAIL: verbosity :: xfail.txt (4 of 5)
# TEST-OUTPUT-OFF-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# TEST-OUTPUT-OFF-NEXT: ********************
# TEST-OUTPUT-OFF-NEXT: Failed Tests (1):
# TEST-OUTPUT-OFF-NEXT:   verbosity :: fail.txt
# TEST-OUTPUT-OFF-EMPTY:
# TEST-OUTPUT-OFF-NEXT: ********************
# TEST-OUTPUT-OFF-NEXT: Unexpectedly Passed Tests (1):
# TEST-OUTPUT-OFF-NEXT:   verbosity :: xpass.txt
# TEST-OUTPUT-OFF-EMPTY:
# TEST-OUTPUT-OFF-EMPTY:
# TEST-OUTPUT-OFF-NEXT: Testing Time: {{.*}}s
# TEST-OUTPUT-OFF-EMPTY:
# TEST-OUTPUT-OFF-NEXT: Total Discovered Tests: 5
# TEST-OUTPUT-OFF-NEXT:   Unsupported        : 1 (20.00%)
# TEST-OUTPUT-OFF-NEXT:   Passed             : 1 (20.00%)
# TEST-OUTPUT-OFF-NEXT:   Expectedly Failed  : 1 (20.00%)
# TEST-OUTPUT-OFF-NEXT:   Failed             : 1 (20.00%)
# TEST-OUTPUT-OFF-NEXT:   Unexpectedly Passed: 1 (20.00%)

# RUN: not %{lit} --test-output all  %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix SHOW-ALL < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt


### Test --print-result-after

# RUN: not %{lit} --print-result-after off  %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix RESULT-OFF < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# RESULT-OFF:      ********************
# RESULT-OFF-NEXT: Failed Tests (1):
# RESULT-OFF-NEXT:   verbosity :: fail.txt
# RESULT-OFF-EMPTY:
# RESULT-OFF-NEXT: ********************
# RESULT-OFF-NEXT: Unexpectedly Passed Tests (1):
# RESULT-OFF-NEXT:   verbosity :: xpass.txt
# RESULT-OFF-EMPTY:
# RESULT-OFF-EMPTY:
# RESULT-OFF-NEXT: Testing Time: {{.*}}s
# RESULT-OFF-EMPTY:
# RESULT-OFF-NEXT: Total Discovered Tests: 5
# RESULT-OFF-NEXT:   Unsupported        : 1 (20.00%)
# RESULT-OFF-NEXT:   Passed             : 1 (20.00%)
# RESULT-OFF-NEXT:   Expectedly Failed  : 1 (20.00%)
# RESULT-OFF-NEXT:   Failed             : 1 (20.00%)
# RESULT-OFF-NEXT:   Unexpectedly Passed: 1 (20.00%)


# RUN: not %{lit} --print-result-after failed  %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix RESULT-FAILED < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# RESULT-FAILED:      -- Testing: 5 tests, 1 workers --
# RESULT-FAILED-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# RESULT-FAILED-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# RESULT-FAILED-NEXT: ********************
# RESULT-FAILED-NEXT: Failed Tests (1):
# RESULT-FAILED-NEXT:   verbosity :: fail.txt
# RESULT-FAILED-EMPTY:
# RESULT-FAILED-NEXT: ********************
# RESULT-FAILED-NEXT: Unexpectedly Passed Tests (1):
# RESULT-FAILED-NEXT:   verbosity :: xpass.txt
# RESULT-FAILED-EMPTY:
# RESULT-FAILED-EMPTY:
# RESULT-FAILED-NEXT: Testing Time: {{.*}}s
# RESULT-FAILED-EMPTY:
# RESULT-FAILED-NEXT: Total Discovered Tests: 5
# RESULT-FAILED-NEXT:   Unsupported        : 1 (20.00%)
# RESULT-FAILED-NEXT:   Passed             : 1 (20.00%)
# RESULT-FAILED-NEXT:   Expectedly Failed  : 1 (20.00%)
# RESULT-FAILED-NEXT:   Failed             : 1 (20.00%)
# RESULT-FAILED-NEXT:   Unexpectedly Passed: 1 (20.00%)


# RUN: not %{lit} --print-result-after all  %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt


### Test combinations of --print-result-after followed by --test-output

# RUN: not %{lit} --print-result-after off --test-output failed %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix RESULT-OFF-OUTPUT-FAILED < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# RESULT-OFF-OUTPUT-FAILED:      -- Testing: 5 tests, 1 workers --
# RESULT-OFF-OUTPUT-FAILED-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# RESULT-OFF-OUTPUT-FAILED-NEXT: ******************** TEST 'verbosity :: fail.txt' FAILED ********************
# RESULT-OFF-OUTPUT-FAILED-NEXT: Exit Code: 127
# RESULT-OFF-OUTPUT-FAILED-EMPTY:
# RESULT-OFF-OUTPUT-FAILED-NEXT: Command Output (stdout):
# RESULT-OFF-OUTPUT-FAILED-NEXT: --
# RESULT-OFF-OUTPUT-FAILED-NEXT: # {{R}}UN: at line 1
# RESULT-OFF-OUTPUT-FAILED-NEXT: echo "fail test output"
# RESULT-OFF-OUTPUT-FAILED-NEXT: # executed command: echo 'fail test output'
# RESULT-OFF-OUTPUT-FAILED-NEXT: # .---command stdout------------
# RESULT-OFF-OUTPUT-FAILED-NEXT: # | fail test output
# RESULT-OFF-OUTPUT-FAILED-NEXT: # `-----------------------------
# RESULT-OFF-OUTPUT-FAILED-NEXT: # {{R}}UN: at line 2
# RESULT-OFF-OUTPUT-FAILED-NEXT: fail
# RESULT-OFF-OUTPUT-FAILED-NEXT: # executed command: fail
# RESULT-OFF-OUTPUT-FAILED-NEXT: # .---command stderr------------
# RESULT-OFF-OUTPUT-FAILED-NEXT: # | 'fail': command not found
# RESULT-OFF-OUTPUT-FAILED-NEXT: # `-----------------------------
# RESULT-OFF-OUTPUT-FAILED-NEXT: # error: command failed with exit status: 127
# RESULT-OFF-OUTPUT-FAILED-EMPTY:
# RESULT-OFF-OUTPUT-FAILED-NEXT: --
# RESULT-OFF-OUTPUT-FAILED-EMPTY:
# RESULT-OFF-OUTPUT-FAILED-NEXT: ********************
# RESULT-OFF-OUTPUT-FAILED-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# RESULT-OFF-OUTPUT-FAILED-NEXT: ******************** TEST 'verbosity :: xpass.txt' FAILED ********************
# RESULT-OFF-OUTPUT-FAILED-NEXT: Exit Code: 0
# RESULT-OFF-OUTPUT-FAILED-EMPTY:
# RESULT-OFF-OUTPUT-FAILED-NEXT: Command Output (stdout):
# RESULT-OFF-OUTPUT-FAILED-NEXT: --
# RESULT-OFF-OUTPUT-FAILED-NEXT: # {{R}}UN: at line 2
# RESULT-OFF-OUTPUT-FAILED-NEXT: echo "xpass test output"
# RESULT-OFF-OUTPUT-FAILED-NEXT: # executed command: echo 'xpass test output'
# RESULT-OFF-OUTPUT-FAILED-NEXT: # .---command stdout------------
# RESULT-OFF-OUTPUT-FAILED-NEXT: # | xpass test output
# RESULT-OFF-OUTPUT-FAILED-NEXT: # `-----------------------------
# RESULT-OFF-OUTPUT-FAILED-EMPTY:
# RESULT-OFF-OUTPUT-FAILED-NEXT: --
# RESULT-OFF-OUTPUT-FAILED-EMPTY:
# RESULT-OFF-OUTPUT-FAILED-NEXT: ********************
# RESULT-OFF-OUTPUT-FAILED-NEXT: ********************
# RESULT-OFF-OUTPUT-FAILED-NEXT: Failed Tests (1):
# RESULT-OFF-OUTPUT-FAILED-NEXT:   verbosity :: fail.txt
# RESULT-OFF-OUTPUT-FAILED-EMPTY:
# RESULT-OFF-OUTPUT-FAILED-NEXT: ********************
# RESULT-OFF-OUTPUT-FAILED-NEXT: Unexpectedly Passed Tests (1):
# RESULT-OFF-OUTPUT-FAILED-NEXT:   verbosity :: xpass.txt
# RESULT-OFF-OUTPUT-FAILED-EMPTY:
# RESULT-OFF-OUTPUT-FAILED-EMPTY:
# RESULT-OFF-OUTPUT-FAILED-NEXT: Testing Time: {{.*}}s
# RESULT-OFF-OUTPUT-FAILED-EMPTY:
# RESULT-OFF-OUTPUT-FAILED-NEXT: Total Discovered Tests: 5
# RESULT-OFF-OUTPUT-FAILED-NEXT:   Unsupported        : 1 (20.00%)
# RESULT-OFF-OUTPUT-FAILED-NEXT:   Passed             : 1 (20.00%)
# RESULT-OFF-OUTPUT-FAILED-NEXT:   Expectedly Failed  : 1 (20.00%)
# RESULT-OFF-OUTPUT-FAILED-NEXT:   Failed             : 1 (20.00%)
# RESULT-OFF-OUTPUT-FAILED-NEXT:   Unexpectedly Passed: 1 (20.00%)

# RUN: not %{lit} --print-result-after all --test-output off %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# RUN: not %{lit} --print-result-after failed --test-output all %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix SHOW-ALL < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt


### Test combinations of --test-output followed by --print-result-after

# RUN: not %{lit} --test-output failed --print-result-after off %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix RESULT-OFF < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# RUN: not %{lit} --test-output off --print-result-after all %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# RUN: not %{lit} --test-output all --print-result-after failed %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix OUTPUT-ALL-RESULT-FAILED < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# OUTPUT-ALL-RESULT-FAILED:      -- Testing: 5 tests, 1 workers --
# OUTPUT-ALL-RESULT-FAILED-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# OUTPUT-ALL-RESULT-FAILED-NEXT: ******************** TEST 'verbosity :: fail.txt' FAILED ********************
# OUTPUT-ALL-RESULT-FAILED-NEXT: Exit Code: 127
# OUTPUT-ALL-RESULT-FAILED-EMPTY:
# OUTPUT-ALL-RESULT-FAILED-NEXT: Command Output (stdout):
# OUTPUT-ALL-RESULT-FAILED-NEXT: --
# OUTPUT-ALL-RESULT-FAILED-NEXT: # {{R}}UN: at line 1
# OUTPUT-ALL-RESULT-FAILED-NEXT: echo "fail test output"
# OUTPUT-ALL-RESULT-FAILED-NEXT: # executed command: echo 'fail test output'
# OUTPUT-ALL-RESULT-FAILED-NEXT: # .---command stdout------------
# OUTPUT-ALL-RESULT-FAILED-NEXT: # | fail test output
# OUTPUT-ALL-RESULT-FAILED-NEXT: # `-----------------------------
# OUTPUT-ALL-RESULT-FAILED-NEXT: # {{R}}UN: at line 2
# OUTPUT-ALL-RESULT-FAILED-NEXT: fail
# OUTPUT-ALL-RESULT-FAILED-NEXT: # executed command: fail
# OUTPUT-ALL-RESULT-FAILED-NEXT: # .---command stderr------------
# OUTPUT-ALL-RESULT-FAILED-NEXT: # | 'fail': command not found
# OUTPUT-ALL-RESULT-FAILED-NEXT: # `-----------------------------
# OUTPUT-ALL-RESULT-FAILED-NEXT: # error: command failed with exit status: 127
# OUTPUT-ALL-RESULT-FAILED-EMPTY:
# OUTPUT-ALL-RESULT-FAILED-NEXT: --
# OUTPUT-ALL-RESULT-FAILED-EMPTY:
# OUTPUT-ALL-RESULT-FAILED-NEXT: ********************
# OUTPUT-ALL-RESULT-FAILED-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# OUTPUT-ALL-RESULT-FAILED-NEXT: ******************** TEST 'verbosity :: xpass.txt' FAILED ********************
# OUTPUT-ALL-RESULT-FAILED-NEXT: Exit Code: 0
# OUTPUT-ALL-RESULT-FAILED-EMPTY:
# OUTPUT-ALL-RESULT-FAILED-NEXT: Command Output (stdout):
# OUTPUT-ALL-RESULT-FAILED-NEXT: --
# OUTPUT-ALL-RESULT-FAILED-NEXT: # {{R}}UN: at line 2
# OUTPUT-ALL-RESULT-FAILED-NEXT: echo "xpass test output"
# OUTPUT-ALL-RESULT-FAILED-NEXT: # executed command: echo 'xpass test output'
# OUTPUT-ALL-RESULT-FAILED-NEXT: # .---command stdout------------
# OUTPUT-ALL-RESULT-FAILED-NEXT: # | xpass test output
# OUTPUT-ALL-RESULT-FAILED-NEXT: # `-----------------------------
# OUTPUT-ALL-RESULT-FAILED-EMPTY:
# OUTPUT-ALL-RESULT-FAILED-NEXT: --
# OUTPUT-ALL-RESULT-FAILED-EMPTY:
# OUTPUT-ALL-RESULT-FAILED-NEXT: ********************
# OUTPUT-ALL-RESULT-FAILED-NEXT: ********************
# OUTPUT-ALL-RESULT-FAILED-NEXT: Failed Tests (1):
# OUTPUT-ALL-RESULT-FAILED-NEXT:   verbosity :: fail.txt
# OUTPUT-ALL-RESULT-FAILED-EMPTY:
# OUTPUT-ALL-RESULT-FAILED-NEXT: ********************
# OUTPUT-ALL-RESULT-FAILED-NEXT: Unexpectedly Passed Tests (1):
# OUTPUT-ALL-RESULT-FAILED-NEXT:   verbosity :: xpass.txt
# OUTPUT-ALL-RESULT-FAILED-EMPTY:
# OUTPUT-ALL-RESULT-FAILED-EMPTY:
# OUTPUT-ALL-RESULT-FAILED-NEXT: Testing Time: {{.*}}
# OUTPUT-ALL-RESULT-FAILED-EMPTY:
# OUTPUT-ALL-RESULT-FAILED-NEXT: Total Discovered Tests: 5
# OUTPUT-ALL-RESULT-FAILED-NEXT:   Unsupported        : 1 (20.00%)
# OUTPUT-ALL-RESULT-FAILED-NEXT:   Passed             : 1 (20.00%)
# OUTPUT-ALL-RESULT-FAILED-NEXT:   Expectedly Failed  : 1 (20.00%)
# OUTPUT-ALL-RESULT-FAILED-NEXT:   Failed             : 1 (20.00%)
# OUTPUT-ALL-RESULT-FAILED-NEXT:   Unexpectedly Passed: 1 (20.00%)


### Test progress bar and terse summary in isolation

# RUN: not %{lit} --progress-bar %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix PROGRESS < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# PROGRESS:      -- Testing: 5 tests, 1 workers --
# PROGRESS-NEXT: Testing:
# PROGRESS-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# PROGRESS-NEXT: Testing:
# PROGRESS-NEXT: PASS: verbosity :: pass.txt (2 of 5)
# PROGRESS-NEXT: Testing:
# PROGRESS-NEXT: {{UN}}SUPPORTED: verbosity :: unsupported.txt (3 of 5)
# PROGRESS-NEXT: Testing:
# PROGRESS-NEXT: {{X}}FAIL: verbosity :: xfail.txt (4 of 5)
# PROGRESS-NEXT: Testing:
# PROGRESS-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# PROGRESS-NEXT: Testing:
# PROGRESS-NEXT: ********************
# PROGRESS-NEXT: Failed Tests (1):
# PROGRESS-NEXT:   verbosity :: fail.txt
# PROGRESS-EMPTY:
# PROGRESS-NEXT: ********************
# PROGRESS-NEXT: Unexpectedly Passed Tests (1):
# PROGRESS-NEXT:   verbosity :: xpass.txt
# PROGRESS-EMPTY:
# PROGRESS-EMPTY:
# PROGRESS-NEXT: Testing Time: {{.*}}s
# PROGRESS-EMPTY:
# PROGRESS-NEXT: Total Discovered Tests: 5
# PROGRESS-NEXT:   Unsupported        : 1 (20.00%)
# PROGRESS-NEXT:   Passed             : 1 (20.00%)
# PROGRESS-NEXT:   Expectedly Failed  : 1 (20.00%)
# PROGRESS-NEXT:   Failed             : 1 (20.00%)
# PROGRESS-NEXT:   Unexpectedly Passed: 1 (20.00%)

# RUN: not %{lit} --terse-summary %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix TERSE < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# TERSE:      -- Testing: 5 tests, 1 workers --
# TERSE-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# TERSE-NEXT: PASS: verbosity :: pass.txt (2 of 5)
# TERSE-NEXT: {{UN}}SUPPORTED: verbosity :: unsupported.txt (3 of 5)
# TERSE-NEXT: {{X}}FAIL: verbosity :: xfail.txt (4 of 5)
# TERSE-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# TERSE-NEXT: ********************
# TERSE-NEXT: Failed Tests (1):
# TERSE-NEXT:   verbosity :: fail.txt
# TERSE-EMPTY:
# TERSE-NEXT: ********************
# TERSE-NEXT: Unexpectedly Passed Tests (1):
# TERSE-NEXT:   verbosity :: xpass.txt
# TERSE-EMPTY:
# TERSE-EMPTY:
# TERSE-NEXT: Total Discovered Tests: 5
# TERSE-NEXT:   Failed             : 1 (20.00%)
# TERSE-NEXT:   Unexpectedly Passed: 1 (20.00%)


### Aliases in combination

# RUN: not %{lit} -a -s %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix AS < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# AS:      -- Testing: 5 tests, 1 workers --
# AS-NEXT: Testing:
# AS-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# AS-NEXT: ******************** TEST 'verbosity :: fail.txt' FAILED ********************
# AS-NEXT: Exit Code: 127
# AS-EMPTY:
# AS-NEXT: Command Output (stdout):
# AS-NEXT: --
# AS-NEXT: # {{R}}UN: at line 1
# AS-NEXT: echo "fail test output"
# AS-NEXT: # executed command: echo 'fail test output'
# AS-NEXT: # .---command stdout------------
# AS-NEXT: # | fail test output
# AS-NEXT: # `-----------------------------
# AS-NEXT: # {{R}}UN: at line 2
# AS-NEXT: fail
# AS-NEXT: # executed command: fail
# AS-NEXT: # .---command stderr------------
# AS-NEXT: # | 'fail': command not found
# AS-NEXT: # `-----------------------------
# AS-NEXT: # error: command failed with exit status: 127
# AS-EMPTY:
# AS-NEXT: --
# AS-EMPTY:
# AS-NEXT: ********************
# AS-NEXT: Testing:
# AS-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# AS-NEXT: ******************** TEST 'verbosity :: xpass.txt' FAILED ********************
# AS-NEXT: Exit Code: 0
# AS-EMPTY:
# AS-NEXT: Command Output (stdout):
# AS-NEXT: --
# AS-NEXT: # {{R}}UN: at line 2
# AS-NEXT: echo "xpass test output"
# AS-NEXT: # executed command: echo 'xpass test output'
# AS-NEXT: # .---command stdout------------
# AS-NEXT: # | xpass test output
# AS-NEXT: # `-----------------------------
# AS-EMPTY:
# AS-NEXT: --
# AS-EMPTY:
# AS-NEXT: ********************
# AS-NEXT: Testing:
# AS-NEXT: ********************
# AS-NEXT: Failed Tests (1):
# AS-NEXT:   verbosity :: fail.txt
# AS-EMPTY:
# AS-NEXT: ********************
# AS-NEXT: Unexpectedly Passed Tests (1):
# AS-NEXT:   verbosity :: xpass.txt
# AS-EMPTY:
# AS-EMPTY:
# AS-NEXT: Testing Time: {{.*}}s
# AS-EMPTY:
# AS-NEXT: Total Discovered Tests: 5
# AS-NEXT:   Unsupported        : 1 (20.00%)
# AS-NEXT:   Passed             : 1 (20.00%)
# AS-NEXT:   Expectedly Failed  : 1 (20.00%)
# AS-NEXT:   Failed             : 1 (20.00%)
# AS-NEXT:   Unexpectedly Passed: 1 (20.00%)


# RUN: not %{lit} -s -a %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix SA < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# SA:      -- Testing: 5 tests, 1 workers --
# SA-NEXT: Testing:
# SA-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# SA-NEXT: ******************** TEST 'verbosity :: fail.txt' FAILED ********************
# SA-NEXT: Exit Code: 127
# SA-EMPTY:
# SA-NEXT: Command Output (stdout):
# SA-NEXT: --
# SA-NEXT: # {{R}}UN: at line 1
# SA-NEXT: echo "fail test output"
# SA-NEXT: # executed command: echo 'fail test output'
# SA-NEXT: # .---command stdout------------
# SA-NEXT: # | fail test output
# SA-NEXT: # `-----------------------------
# SA-NEXT: # {{R}}UN: at line 2
# SA-NEXT: fail
# SA-NEXT: # executed command: fail
# SA-NEXT: # .---command stderr------------
# SA-NEXT: # | 'fail': command not found
# SA-NEXT: # `-----------------------------
# SA-NEXT: # error: command failed with exit status: 127
# SA-EMPTY:
# SA-NEXT: --
# SA-EMPTY:
# SA-NEXT: ********************
# SA-NEXT: Testing:
# SA-NEXT: PASS: verbosity :: pass.txt (2 of 5)
# SA-NEXT: Exit Code: 0
# SA-EMPTY:
# SA-NEXT: Command Output (stdout):
# SA-NEXT: --
# SA-NEXT: # {{R}}UN: at line 1
# SA-NEXT: echo "pass test output"
# SA-NEXT: # executed command: echo 'pass test output'
# SA-NEXT: # .---command stdout------------
# SA-NEXT: # | pass test output
# SA-NEXT: # `-----------------------------
# SA-EMPTY:
# SA-NEXT: --
# SA-EMPTY:
# SA-NEXT: ********************
# SA-NEXT: Testing:
# SA-NEXT: {{UN}}SUPPORTED: verbosity :: unsupported.txt (3 of 5)
# SA-NEXT: Test requires the following unavailable features: asdf
# SA-NEXT: ********************
# SA-NEXT: Testing:
# SA-NEXT: {{X}}FAIL: verbosity :: xfail.txt (4 of 5)
# SA-NEXT: Exit Code: 1
# SA-EMPTY:
# SA-NEXT: Command Output (stdout):
# SA-NEXT: --
# SA-NEXT: # {{R}}UN: at line 2
# SA-NEXT: not echo "xfail test output"
# SA-NEXT: # executed command: not echo 'xfail test output'
# SA-NEXT: # .---command stdout------------
# SA-NEXT: # | xfail test output
# SA-NEXT: # `-----------------------------
# SA-NEXT: # error: command failed with exit status: 1
# SA-EMPTY:
# SA-NEXT: --
# SA-EMPTY:
# SA-NEXT: ********************
# SA-NEXT: Testing:
# SA-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# SA-NEXT: ******************** TEST 'verbosity :: xpass.txt' FAILED ********************
# SA-NEXT: Exit Code: 0
# SA-EMPTY:
# SA-NEXT: Command Output (stdout):
# SA-NEXT: --
# SA-NEXT: # {{R}}UN: at line 2
# SA-NEXT: echo "xpass test output"
# SA-NEXT: # executed command: echo 'xpass test output'
# SA-NEXT: # .---command stdout------------
# SA-NEXT: # | xpass test output
# SA-NEXT: # `-----------------------------
# SA-EMPTY:
# SA-NEXT: --
# SA-EMPTY:
# SA-NEXT: ********************
# SA-NEXT: Testing:
# SA-NEXT: ********************
# SA-NEXT: Failed Tests (1):
# SA-NEXT:   verbosity :: fail.txt
# SA-EMPTY:
# SA-NEXT: ********************
# SA-NEXT: Unexpectedly Passed Tests (1):
# SA-NEXT:   verbosity :: xpass.txt
# SA-EMPTY:
# SA-EMPTY:
# SA-NEXT: Testing Time: {{.*}}s
# SA-EMPTY:
# SA-NEXT: Total Discovered Tests: 5
# SA-NEXT:   Unsupported        : 1 (20.00%)
# SA-NEXT:   Passed             : 1 (20.00%)
# SA-NEXT:   Expectedly Failed  : 1 (20.00%)
# SA-NEXT:   Failed             : 1 (20.00%)
# SA-NEXT:   Unexpectedly Passed: 1 (20.00%)


# RUN: not %{lit} -q -a %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix QA < %t/stdout.txt
# RUN: FileCheck %s --check-prefix QUIET-ERR --implicit-check-not lit < %t/stderr.txt

# QA:      -- Testing: 5 tests, 1 workers --
# QA-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# QA-NEXT: ******************** TEST 'verbosity :: fail.txt' FAILED ********************
# QA-NEXT: Exit Code: 127
# QA-EMPTY:
# QA-NEXT: Command Output (stdout):
# QA-NEXT: --
# QA-NEXT: # {{R}}UN: at line 1
# QA-NEXT: echo "fail test output"
# QA-NEXT: # executed command: echo 'fail test output'
# QA-NEXT: # .---command stdout------------
# QA-NEXT: # | fail test output
# QA-NEXT: # `-----------------------------
# QA-NEXT: # {{R}}UN: at line 2
# QA-NEXT: fail
# QA-NEXT: # executed command: fail
# QA-NEXT: # .---command stderr------------
# QA-NEXT: # | 'fail': command not found
# QA-NEXT: # `-----------------------------
# QA-NEXT: # error: command failed with exit status: 127
# QA-EMPTY:
# QA-NEXT: --
# QA-EMPTY:
# QA-NEXT: ********************
# QA-NEXT: PASS: verbosity :: pass.txt (2 of 5)
# QA-NEXT: Exit Code: 0
# QA-EMPTY:
# QA-NEXT: Command Output (stdout):
# QA-NEXT: --
# QA-NEXT: # {{R}}UN: at line 1
# QA-NEXT: echo "pass test output"
# QA-NEXT: # executed command: echo 'pass test output'
# QA-NEXT: # .---command stdout------------
# QA-NEXT: # | pass test output
# QA-NEXT: # `-----------------------------
# QA-EMPTY:
# QA-NEXT: --
# QA-EMPTY:
# QA-NEXT: ********************
# QA-NEXT: {{UN}}SUPPORTED: verbosity :: unsupported.txt (3 of 5)
# QA-NEXT: Test requires the following unavailable features: asdf
# QA-NEXT: ********************
# QA-NEXT: {{X}}FAIL: verbosity :: xfail.txt (4 of 5)
# QA-NEXT: Exit Code: 1
# QA-EMPTY:
# QA-NEXT: Command Output (stdout):
# QA-NEXT: --
# QA-NEXT: # {{R}}UN: at line 2
# QA-NEXT: not echo "xfail test output"
# QA-NEXT: # executed command: not echo 'xfail test output'
# QA-NEXT: # .---command stdout------------
# QA-NEXT: # | xfail test output
# QA-NEXT: # `-----------------------------
# QA-NEXT: # error: command failed with exit status: 1
# QA-EMPTY:
# QA-NEXT: --
# QA-EMPTY:
# QA-NEXT: ********************
# QA-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# QA-NEXT: ******************** TEST 'verbosity :: xpass.txt' FAILED ********************
# QA-NEXT: Exit Code: 0
# QA-EMPTY:
# QA-NEXT: Command Output (stdout):
# QA-NEXT: --
# QA-NEXT: # {{R}}UN: at line 2
# QA-NEXT: echo "xpass test output"
# QA-NEXT: # executed command: echo 'xpass test output'
# QA-NEXT: # .---command stdout------------
# QA-NEXT: # | xpass test output
# QA-NEXT: # `-----------------------------
# QA-EMPTY:
# QA-NEXT: --
# QA-EMPTY:
# QA-NEXT: ********************
# QA-NEXT: ********************
# QA-NEXT: Failed Tests (1):
# QA-NEXT:   verbosity :: fail.txt
# QA-EMPTY:
# QA-NEXT: ********************
# QA-NEXT: Unexpectedly Passed Tests (1):
# QA-NEXT:   verbosity :: xpass.txt
# QA-EMPTY:
# QA-EMPTY:
# QA-NEXT: Total Discovered Tests: 5
# QA-NEXT:   Failed             : 1 (20.00%)
# QA-NEXT:   Unexpectedly Passed: 1 (20.00%)

# RUN: not %{lit} -a -q %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix QUIET < %t/stdout.txt
# RUN: FileCheck %s --check-prefix QUIET-ERR --implicit-check-not lit < %t/stderr.txt

# RUN: not %{lit} -sqav %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix SQAV < %t/stdout.txt
# RUN: FileCheck %s --check-prefix QUIET-ERR --implicit-check-not lit < %t/stderr.txt

# SQAV:      -- Testing: 5 tests, 1 workers --
# SQAV-NEXT: Testing:
# SQAV-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# SQAV-NEXT: ******************** TEST 'verbosity :: fail.txt' FAILED ********************
# SQAV-NEXT: Exit Code: 127
# SQAV-EMPTY:
# SQAV-NEXT: Command Output (stdout):
# SQAV-NEXT: --
# SQAV-NEXT: # {{R}}UN: at line 1
# SQAV-NEXT: echo "fail test output"
# SQAV-NEXT: # executed command: echo 'fail test output'
# SQAV-NEXT: # .---command stdout------------
# SQAV-NEXT: # | fail test output
# SQAV-NEXT: # `-----------------------------
# SQAV-NEXT: # {{R}}UN: at line 2
# SQAV-NEXT: fail
# SQAV-NEXT: # executed command: fail
# SQAV-NEXT: # .---command stderr------------
# SQAV-NEXT: # | 'fail': command not found
# SQAV-NEXT: # `-----------------------------
# SQAV-NEXT: # error: command failed with exit status: 127
# SQAV-EMPTY:
# SQAV-NEXT: --
# SQAV-EMPTY:
# SQAV-NEXT: ********************
# SQAV-NEXT: Testing:
# SQAV-NEXT: PASS: verbosity :: pass.txt (2 of 5)
# SQAV-NEXT: Testing:
# SQAV-NEXT: {{UN}}SUPPORTED: verbosity :: unsupported.txt (3 of 5)
# SQAV-NEXT: Testing:
# SQAV-NEXT: {{X}}FAIL: verbosity :: xfail.txt (4 of 5)
# SQAV-NEXT: Testing:
# SQAV-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# SQAV-NEXT: ******************** TEST 'verbosity :: xpass.txt' FAILED ********************
# SQAV-NEXT: Exit Code: 0
# SQAV-EMPTY:
# SQAV-NEXT: Command Output (stdout):
# SQAV-NEXT: --
# SQAV-NEXT: # {{R}}UN: at line 2
# SQAV-NEXT: echo "xpass test output"
# SQAV-NEXT: # executed command: echo 'xpass test output'
# SQAV-NEXT: # .---command stdout------------
# SQAV-NEXT: # | xpass test output
# SQAV-NEXT: # `-----------------------------
# SQAV-EMPTY:
# SQAV-NEXT: --
# SQAV-EMPTY:
# SQAV-NEXT: ********************
# SQAV-NEXT: Testing:
# SQAV-NEXT: ********************
# SQAV-NEXT: Failed Tests (1):
# SQAV-NEXT:   verbosity :: fail.txt
# SQAV-EMPTY:
# SQAV-NEXT: ********************
# SQAV-NEXT: Unexpectedly Passed Tests (1):
# SQAV-NEXT:   verbosity :: xpass.txt
# SQAV-EMPTY:
# SQAV-EMPTY:
# SQAV-NEXT: Total Discovered Tests: 5
# SQAV-NEXT:   Failed             : 1 (20.00%)
# SQAV-NEXT:   Unexpectedly Passed: 1 (20.00%)


### Aliases with specific overrides

# RUN: not %{lit} --quiet --no-terse-summary %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix QUIET-W-SUMMARY < %t/stdout.txt
# RUN: FileCheck %s --check-prefix QUIET-ERR --implicit-check-not lit < %t/stderr.txt

# QUIET-W-SUMMARY:      -- Testing: 5 tests, 1 workers --
# QUIET-W-SUMMARY-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# QUIET-W-SUMMARY-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# QUIET-W-SUMMARY-NEXT: ********************
# QUIET-W-SUMMARY-NEXT: Failed Tests (1):
# QUIET-W-SUMMARY-NEXT:   verbosity :: fail.txt
# QUIET-W-SUMMARY-EMPTY:
# QUIET-W-SUMMARY-NEXT: ********************
# QUIET-W-SUMMARY-NEXT: Unexpectedly Passed Tests (1):
# QUIET-W-SUMMARY-NEXT:   verbosity :: xpass.txt
# QUIET-W-SUMMARY-EMPTY:
# QUIET-W-SUMMARY-EMPTY:
# QUIET-W-SUMMARY-NEXT: Testing Time: {{.*}}s
# QUIET-W-SUMMARY-EMPTY:
# QUIET-W-SUMMARY-NEXT: Total Discovered Tests: 5
# QUIET-W-SUMMARY-NEXT:   Unsupported        : 1 (20.00%)
# QUIET-W-SUMMARY-NEXT:   Passed             : 1 (20.00%)
# QUIET-W-SUMMARY-NEXT:   Expectedly Failed  : 1 (20.00%)
# QUIET-W-SUMMARY-NEXT:   Failed             : 1 (20.00%)
# QUIET-W-SUMMARY-NEXT:   Unexpectedly Passed: 1 (20.00%)


# RUN: not %{lit} --quiet --progress-bar %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix QUIET-W-PROGRESS < %t/stdout.txt
# RUN: FileCheck %s --check-prefix QUIET-ERR --implicit-check-not lit < %t/stderr.txt

# QUIET-W-PROGRESS: -- Testing: 5 tests, 1 workers --
# QUIET-W-PROGRESS-NEXT: Testing:
# QUIET-W-PROGRESS-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# QUIET-W-PROGRESS-NEXT: Testing:
# QUIET-W-PROGRESS-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# QUIET-W-PROGRESS-NEXT: Testing:
# QUIET-W-PROGRESS-NEXT: ********************
# QUIET-W-PROGRESS-NEXT: Failed Tests (1):
# QUIET-W-PROGRESS-NEXT:   verbosity :: fail.txt
# QUIET-W-PROGRESS-EMPTY:
# QUIET-W-PROGRESS-NEXT: ********************
# QUIET-W-PROGRESS-NEXT: Unexpectedly Passed Tests (1):
# QUIET-W-PROGRESS-NEXT:   verbosity :: xpass.txt
# QUIET-W-PROGRESS-EMPTY:
# QUIET-W-PROGRESS-EMPTY:
# QUIET-W-PROGRESS-NEXT: Total Discovered Tests: 5
# QUIET-W-PROGRESS-NEXT:   Failed             : 1 (20.00%)
# QUIET-W-PROGRESS-NEXT:   Unexpectedly Passed: 1 (20.00%)

# RUN: not %{lit} --show-all --terse-summary %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix ALL-TERSE < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# ALL-TERSE: -- Testing: 5 tests, 1 workers --
# ALL-TERSE-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# ALL-TERSE-NEXT: ******************** TEST 'verbosity :: fail.txt' FAILED ********************
# ALL-TERSE-NEXT: Exit Code: 127
# ALL-TERSE-EMPTY:
# ALL-TERSE-NEXT: Command Output (stdout):
# ALL-TERSE-NEXT: --
# ALL-TERSE-NEXT: # {{R}}UN: at line 1
# ALL-TERSE-NEXT: echo "fail test output"
# ALL-TERSE-NEXT: # executed command: echo 'fail test output'
# ALL-TERSE-NEXT: # .---command stdout------------
# ALL-TERSE-NEXT: # | fail test output
# ALL-TERSE-NEXT: # `-----------------------------
# ALL-TERSE-NEXT: # {{R}}UN: at line 2
# ALL-TERSE-NEXT: fail
# ALL-TERSE-NEXT: # executed command: fail
# ALL-TERSE-NEXT: # .---command stderr------------
# ALL-TERSE-NEXT: # | 'fail': command not found
# ALL-TERSE-NEXT: # `-----------------------------
# ALL-TERSE-NEXT: # error: command failed with exit status: 127
# ALL-TERSE-EMPTY:
# ALL-TERSE-NEXT: --
# ALL-TERSE-EMPTY:
# ALL-TERSE-NEXT: ********************
# ALL-TERSE-NEXT: PASS: verbosity :: pass.txt (2 of 5)
# ALL-TERSE-NEXT: Exit Code: 0
# ALL-TERSE-EMPTY:
# ALL-TERSE-NEXT: Command Output (stdout):
# ALL-TERSE-NEXT: --
# ALL-TERSE-NEXT: # {{R}}UN: at line 1
# ALL-TERSE-NEXT: echo "pass test output"
# ALL-TERSE-NEXT: # executed command: echo 'pass test output'
# ALL-TERSE-NEXT: # .---command stdout------------
# ALL-TERSE-NEXT: # | pass test output
# ALL-TERSE-NEXT: # `-----------------------------
# ALL-TERSE-EMPTY:
# ALL-TERSE-NEXT: --
# ALL-TERSE-EMPTY:
# ALL-TERSE-NEXT: ********************
# ALL-TERSE-NEXT: {{UN}}SUPPORTED: verbosity :: unsupported.txt (3 of 5)
# ALL-TERSE-NEXT: Test requires the following unavailable features: asdf
# ALL-TERSE-NEXT: ********************
# ALL-TERSE-NEXT: {{X}}FAIL: verbosity :: xfail.txt (4 of 5)
# ALL-TERSE-NEXT: Exit Code: 1
# ALL-TERSE-EMPTY:
# ALL-TERSE-NEXT: Command Output (stdout):
# ALL-TERSE-NEXT: --
# ALL-TERSE-NEXT: # {{R}}UN: at line 2
# ALL-TERSE-NEXT: not echo "xfail test output"
# ALL-TERSE-NEXT: # executed command: not echo 'xfail test output'
# ALL-TERSE-NEXT: # .---command stdout------------
# ALL-TERSE-NEXT: # | xfail test output
# ALL-TERSE-NEXT: # `-----------------------------
# ALL-TERSE-NEXT: # error: command failed with exit status: 1
# ALL-TERSE-EMPTY:
# ALL-TERSE-NEXT: --
# ALL-TERSE-EMPTY:
# ALL-TERSE-NEXT: ********************
# ALL-TERSE-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# ALL-TERSE-NEXT: ******************** TEST 'verbosity :: xpass.txt' FAILED ********************
# ALL-TERSE-NEXT: Exit Code: 0
# ALL-TERSE-EMPTY:
# ALL-TERSE-NEXT: Command Output (stdout):
# ALL-TERSE-NEXT: --
# ALL-TERSE-NEXT: # {{R}}UN: at line 2
# ALL-TERSE-NEXT: echo "xpass test output"
# ALL-TERSE-NEXT: # executed command: echo 'xpass test output'
# ALL-TERSE-NEXT: # .---command stdout------------
# ALL-TERSE-NEXT: # | xpass test output
# ALL-TERSE-NEXT: # `-----------------------------
# ALL-TERSE-EMPTY:
# ALL-TERSE-NEXT: --
# ALL-TERSE-EMPTY:
# ALL-TERSE-NEXT: ********************
# ALL-TERSE-NEXT: ********************
# ALL-TERSE-NEXT: Failed Tests (1):
# ALL-TERSE-NEXT:   verbosity :: fail.txt
# ALL-TERSE-EMPTY:
# ALL-TERSE-NEXT: ********************
# ALL-TERSE-NEXT: Unexpectedly Passed Tests (1):
# ALL-TERSE-NEXT:   verbosity :: xpass.txt
# ALL-TERSE-EMPTY:
# ALL-TERSE-EMPTY:
# ALL-TERSE-NEXT: Total Discovered Tests: 5
# ALL-TERSE-NEXT:   Failed             : 1 (20.00%)
# ALL-TERSE-NEXT:   Unexpectedly Passed: 1 (20.00%)

# RUN: not %{lit} --show-all --diagnostic-level error %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix SHOW-ALL < %t/stdout.txt
# RUN: FileCheck %s --check-prefix QUIET-ERR --implicit-check-not lit < %t/stderr.txt

# RUN: not %{lit} --show-all --test-output off %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# RUN: not %{lit} --succinct --print-result-after all %{inputs}/verbosity 2> %t/stderr.txt > %t/stdout.txt
# RUN: FileCheck %s --check-prefix SUCCINCT-RESULT-ALL < %t/stdout.txt
# RUN: FileCheck %s --check-prefix NO-ARGS-ERR --implicit-check-not lit < %t/stderr.txt

# SUCCINCT-RESULT-ALL:      -- Testing: 5 tests, 1 workers --
# SUCCINCT-RESULT-ALL-NEXT: Testing:
# SUCCINCT-RESULT-ALL-NEXT: FAIL: verbosity :: fail.txt (1 of 5)
# SUCCINCT-RESULT-ALL-NEXT: Testing:
# SUCCINCT-RESULT-ALL-NEXT: PASS: verbosity :: pass.txt (2 of 5)
# SUCCINCT-RESULT-ALL-NEXT: Testing:
# SUCCINCT-RESULT-ALL-NEXT: {{UN}}SUPPORTED: verbosity :: unsupported.txt (3 of 5)
# SUCCINCT-RESULT-ALL-NEXT: Testing:
# SUCCINCT-RESULT-ALL-NEXT: {{X}}FAIL: verbosity :: xfail.txt (4 of 5)
# SUCCINCT-RESULT-ALL-NEXT: Testing:
# SUCCINCT-RESULT-ALL-NEXT: XPASS: verbosity :: xpass.txt (5 of 5)
# SUCCINCT-RESULT-ALL-NEXT: Testing:
# SUCCINCT-RESULT-ALL-NEXT: ********************
# SUCCINCT-RESULT-ALL-NEXT: Failed Tests (1):
# SUCCINCT-RESULT-ALL-NEXT:   verbosity :: fail.txt
# SUCCINCT-RESULT-ALL-EMPTY:
# SUCCINCT-RESULT-ALL-NEXT: ********************
# SUCCINCT-RESULT-ALL-NEXT: Unexpectedly Passed Tests (1):
# SUCCINCT-RESULT-ALL-NEXT:   verbosity :: xpass.txt
# SUCCINCT-RESULT-ALL-EMPTY:
# SUCCINCT-RESULT-ALL-EMPTY:
# SUCCINCT-RESULT-ALL-NEXT: Testing Time: {{.*}}s
# SUCCINCT-RESULT-ALL-EMPTY:
# SUCCINCT-RESULT-ALL-NEXT: Total Discovered Tests: 5
# SUCCINCT-RESULT-ALL-NEXT:   Unsupported        : 1 (20.00%)
# SUCCINCT-RESULT-ALL-NEXT:   Passed             : 1 (20.00%)
# SUCCINCT-RESULT-ALL-NEXT:   Expectedly Failed  : 1 (20.00%)
# SUCCINCT-RESULT-ALL-NEXT:   Failed             : 1 (20.00%)
# SUCCINCT-RESULT-ALL-NEXT:   Unexpectedly Passed: 1 (20.00%)
