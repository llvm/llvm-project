# Check the various features of the ShTest format.

## This is required for the use of %errc_ENOENT.
# REQUIRES: llvm-config-available

# RUN: rm -f %t.xml
# RUN: not %{lit} -v %{inputs}/shtest-format --xunit-xml-output %t.xml > %t.out
# RUN: cp %t.xml /tmp/test
# RUN: FileCheck -DERROR_MSG=%errc_ENOENT < %t.out %s
# RUN: FileCheck --check-prefix=XUNIT < %t.xml %s

# END.

# CHECK: -- Testing:

#       CHECK: FAIL: shtest-format :: fail.txt
#  CHECK-NEXT: *** TEST 'shtest-format :: fail.txt' FAILED ***
#  CHECK-NEXT: Exit Code: 1
# CHECK-EMPTY:
#  CHECK-NEXT: Command Output (stdout):
#  CHECK-NEXT: --
#  CHECK-NEXT: # RUN: at line 1
#  CHECK-NEXT: printf "line 1: failed test output on stdout\nline 2: failed test output on stdout"
#  CHECK-NEXT: executed command: printf 'line 1: failed test output on stdout\nline 2: failed test output on stdout'
#  CHECK-NEXT: # .---command stdout------------
#  CHECK-NEXT: # | line 1: failed test output on stdout
#  CHECK-NEXT: # | line 2: failed test output on stdout
#  CHECK-NEXT: # `-----------------------------
#  CHECK-NEXT: # RUN: at line 2
#  CHECK-NEXT: false
#  CHECK-NEXT: # executed command: false
#  CHECK-NEXT: # note: command had no output on stdout or stderr
#  CHECK-NEXT: # error: command failed with exit status: 1
# CHECK-EMPTY:
#  CHECK-NEXT: --

# CHECK: FAIL: shtest-format :: fail_with_bad_encoding.txt
# CHECK-NEXT: *** TEST 'shtest-format :: fail_with_bad_encoding.txt' FAILED ***
# CHECK: .---command stdout------------
# CHECK-NEXT: a line with bad encoding:
# CHECK: --

# CHECK: FAIL: shtest-format :: fail_with_control_chars.txt
# CHECK-NEXT: *** TEST 'shtest-format :: fail_with_control_chars.txt' FAILED ***
# CHECK: .---command stdout------------
# CHECK-NEXT: a line with {{.*}}control characters{{.*}}.
# CHECK: --

# CHECK: UNRESOLVED: shtest-format :: no-test-line.txt
# CHECK: PASS: shtest-format :: pass.txt
# CHECK: UNSUPPORTED: shtest-format :: requires-missing.txt
# CHECK: PASS: shtest-format :: requires-present.txt
# CHECK: UNRESOLVED: shtest-format :: requires-star.txt
# CHECK: PASS: shtest-format :: requires-triple.txt
# CHECK: PASS: shtest-format :: unsupported-expr-false.txt
# CHECK: UNSUPPORTED: shtest-format :: unsupported-expr-true.txt
# CHECK: UNRESOLVED: shtest-format :: unsupported-star.txt
# CHECK: UNSUPPORTED: shtest-format :: unsupported_dir/some-test.txt
# CHECK: PASS: shtest-format :: xfail-expr-false.txt
# CHECK: XFAIL: shtest-format :: xfail-expr-true.txt
# CHECK: XFAIL: shtest-format :: xfail-feature.txt
# CHECK: XFAIL: shtest-format :: xfail-target.txt
# CHECK: XFAIL: shtest-format :: xfail.txt

#       CHECK: XPASS: shtest-format :: xpass.txt
#  CHECK-NEXT: *** TEST 'shtest-format :: xpass.txt' FAILED ***
#  CHECK-NEXT: Exit Code: 0
# CHECK-EMPTY:
#  CHECK-NEXT: Command Output (stdout):
#  CHECK-NEXT: --
#  CHECK-NEXT: # RUN: at line 1
#  CHECK-NEXT: true
#  CHECK-NEXT: # executed command: true
# CHECK-EMPTY:
#  CHECK-NEXT: --

# CHECK: Failed Tests (3)
# CHECK: shtest-format :: fail.txt
# CHECK: shtest-format :: fail_with_bad_encoding.txt
# CHECK: shtest-format :: fail_with_control_chars.txt

# CHECK: Unexpectedly Passed Tests (1)
# CHECK: shtest-format :: xpass.txt

# CHECK: Testing Time:
# CHECK: Unsupported        : 3
# CHECK: Passed             : 6
# CHECK: Expectedly Failed  : 4
# CHECK: Unresolved         : 3
# CHECK: Failed             : 3
# CHECK: Unexpectedly Passed: 1


# XUNIT: <?xml version="1.0" encoding="UTF-8"?>
# XUNIT-NEXT: <testsuites time="{{[0-9.]+}}">
# XUNIT-NEXT: <testsuite name="shtest-format" tests="20" failures="7" skipped="3" time="{{[0-9.]+}}">

# XUNIT: <testcase classname="shtest-format.shtest-format" name="fail.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT: <failure{{[ ]*}}>
# XUNIT: </failure>
# XUNIT-NEXT: </testcase>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="fail_with_bad_encoding.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT: <failure{{[ ]*}}>
# XUNIT: </failure>
# XUNIT-NEXT: </testcase>

#       XUNIT: <testcase classname="shtest-format.shtest-format" name="fail_with_control_chars.txt" time="{{[0-9]+\.[0-9]+}}">
#  XUNIT-NEXT: <failure><![CDATA[Exit Code: 1
# XUNIT-EMPTY:
#       XUNIT: .---command stdout------------
#  XUNIT-NEXT: a line with [2;30;41mcontrol characters[0m.
#  XUNIT-NEXT: --
#       XUNIT: ]]></failure>
#  XUNIT-NEXT: </testcase>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="no-test-line.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT: <failure{{[ ]*}}>
# XUNIT: </failure>
# XUNIT-NEXT: </testcase>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="pass.txt" time="{{[0-9]+\.[0-9]+}}"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="requires-missing.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT:<skipped message="Missing required feature(s): a-missing-feature"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="requires-present.txt" time="{{[0-9]+\.[0-9]+}}"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="requires-star.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT: <failure{{[ ]*}}>
# XUNIT: </failure>
# XUNIT-NEXT: </testcase>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="requires-triple.txt" time="{{[0-9]+\.[0-9]+}}"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="unsupported-expr-false.txt" time="{{[0-9]+\.[0-9]+}}"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="unsupported-expr-true.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT:<skipped message="Unsupported configuration"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="unsupported-star.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT: <failure{{[ ]*}}>
# XUNIT: </failure>
# XUNIT-NEXT: </testcase>

# XUNIT: <testcase classname="shtest-format.unsupported_dir" name="some-test.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT:<skipped message="Unsupported configuration"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="xfail-expr-false.txt" time="{{[0-9]+\.[0-9]+}}"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="xfail-expr-true.txt" time="{{[0-9]+\.[0-9]+}}"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="xfail-feature.txt" time="{{[0-9]+\.[0-9]+}}"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="xfail-target.txt" time="{{[0-9]+\.[0-9]+}}"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="xfail.txt" time="{{[0-9]+\.[0-9]+}}"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="xpass.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT: <failure{{[ ]*}}>
# XUNIT: </failure>
# XUNIT-NEXT: </testcase>

# XUNIT: </testsuite>
# XUNIT-NEXT: </testsuites>
