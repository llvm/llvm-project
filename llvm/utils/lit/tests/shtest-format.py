# Check the various features of the ShTest format.
#
# RUN: rm -f %t.xml
# RUN: not %{lit} -j 1 -v %{inputs}/shtest-format --xunit-xml-output %t.xml > %t.out
# RUN: FileCheck < %t.out %s
# RUN: FileCheck --check-prefix=XUNIT < %t.xml %s

# END.

# CHECK: -- Testing:

# CHECK: PASS: shtest-format :: argv0.txt
# CHECK: FAIL: shtest-format :: external_shell/fail.txt
# CHECK-NEXT: *** TEST 'shtest-format :: external_shell/fail.txt' FAILED ***
# CHECK: Command Output (stdout):
# CHECK-NEXT: --
# CHECK-NEXT: line 1: failed test output on stdout
# CHECK-NEXT: line 2: failed test output on stdout
# CHECK: Command Output (stderr):
# CHECK-NEXT: --
# CHECK-NEXT: cat{{(\.exe)?}}: {{cannot open does-not-exist|does-not-exist: No such file or directory}}
# CHECK: --

# CHECK: FAIL: shtest-format :: external_shell/fail_with_bad_encoding.txt
# CHECK-NEXT: *** TEST 'shtest-format :: external_shell/fail_with_bad_encoding.txt' FAILED ***
# CHECK: Command Output (stdout):
# CHECK-NEXT: --
# CHECK-NEXT: a line with bad encoding:
# CHECK: --

# CHECK: PASS: shtest-format :: external_shell/pass.txt

# CHECK: FAIL: shtest-format :: fail.txt
# CHECK-NEXT: *** TEST 'shtest-format :: fail.txt' FAILED ***
# CHECK-NEXT: Script:
# CHECK-NEXT: --
# CHECK-NEXT: printf "line 1
# CHECK-NEXT: false
# CHECK-NEXT: --
# CHECK-NEXT: Exit Code: 1
#
# CHECK: Command Output (stdout):
# CHECK-NEXT: --
# CHECK-NEXT: $ ":" "RUN: at line 1"
# CHECK-NEXT: $ "printf"
# CHECK-NEXT: # command output:
# CHECK-NEXT: line 1: failed test output on stdout
# CHECK-NEXT: line 2: failed test output on stdout

# CHECK: UNRESOLVED: shtest-format :: no-test-line.txt
# CHECK: PASS: shtest-format :: pass.txt
# CHECK: UNSUPPORTED: shtest-format :: requires-any-missing.txt
# CHECK: PASS: shtest-format :: requires-any-present.txt
# CHECK: UNSUPPORTED: shtest-format :: requires-missing.txt
# CHECK: PASS: shtest-format :: requires-present.txt
# CHECK: UNRESOLVED: shtest-format :: requires-star.txt
# CHECK: UNSUPPORTED: shtest-format :: requires-triple.txt
# CHECK: PASS: shtest-format :: unsupported-expr-false.txt
# CHECK: UNSUPPORTED: shtest-format :: unsupported-expr-true.txt
# CHECK: UNRESOLVED: shtest-format :: unsupported-star.txt
# CHECK: UNSUPPORTED: shtest-format :: unsupported_dir/some-test.txt
# CHECK: PASS: shtest-format :: xfail-expr-false.txt
# CHECK: XFAIL: shtest-format :: xfail-expr-true.txt
# CHECK: XFAIL: shtest-format :: xfail-feature.txt
# CHECK: XFAIL: shtest-format :: xfail-target.txt
# CHECK: XFAIL: shtest-format :: xfail.txt
# CHECK: XPASS: shtest-format :: xpass.txt
# CHECK-NEXT: *** TEST 'shtest-format :: xpass.txt' FAILED ***
# CHECK-NEXT: Script
# CHECK-NEXT: --
# CHECK-NEXT: true
# CHECK-NEXT: --
# CHECK: Testing Time

# CHECK: Unexpected Passing Tests (1)
# CHECK: shtest-format :: xpass.txt

# CHECK: Failing Tests (3)
# CHECK: shtest-format :: external_shell/fail.txt
# CHECK: shtest-format :: external_shell/fail_with_bad_encoding.txt
# CHECK: shtest-format :: fail.txt

# CHECK: Expected Passes    : 7
# CHECK: Expected Failures  : 4
# CHECK: Unsupported Tests  : 5
# CHECK: Unresolved Tests   : 3
# CHECK: Unexpected Passes  : 1
# CHECK: Unexpected Failures: 3


# XUNIT: <?xml version="1.0" encoding="UTF-8" ?>
# XUNIT-NEXT: <testsuites>
# XUNIT-NEXT: <testsuite name="shtest-format" tests="23" failures="7" skipped="5">

# XUNIT: <testcase classname="shtest-format.shtest-format" name="argv0.txt" time="{{[0-9]+\.[0-9]+}}"/>

# XUNIT: <testcase classname="shtest-format.external_shell" name="fail.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT: <failure{{[ ]*}}>
# XUNIT: </failure>
# XUNIT-NEXT: </testcase>


# XUNIT: <testcase classname="shtest-format.external_shell" name="fail_with_bad_encoding.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT: <failure{{[ ]*}}>
# XUNIT: </failure>
# XUNIT-NEXT: </testcase>

# XUNIT: <testcase classname="shtest-format.external_shell" name="pass.txt" time="{{[0-9]+\.[0-9]+}}"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="fail.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT: <failure{{[ ]*}}>
# XUNIT: </failure>
# XUNIT-NEXT: </testcase>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="no-test-line.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT: <failure{{[ ]*}}>
# XUNIT: </failure>
# XUNIT-NEXT: </testcase>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="pass.txt" time="{{[0-9]+\.[0-9]+}}"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="requires-any-missing.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT:<skipped message="Skipping because of: a-missing-feature || a-missing-feature-2" />

# XUNIT: <testcase classname="shtest-format.shtest-format" name="requires-any-present.txt" time="{{[0-9]+\.[0-9]+}}"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="requires-missing.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT:<skipped message="Skipping because of: a-missing-feature" />

# XUNIT: <testcase classname="shtest-format.shtest-format" name="requires-present.txt" time="{{[0-9]+\.[0-9]+}}"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="requires-star.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT: <failure{{[ ]*}}>
# XUNIT: </failure>
# XUNIT-NEXT: </testcase>


# XUNIT: <testcase classname="shtest-format.shtest-format" name="requires-triple.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT:<skipped message="Skipping because of: x86_64" />

# XUNIT: <testcase classname="shtest-format.shtest-format" name="unsupported-expr-false.txt" time="{{[0-9]+\.[0-9]+}}"/>

# XUNIT: <testcase classname="shtest-format.shtest-format" name="unsupported-expr-true.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT:<skipped message="Skipping because of configuration." />

# XUNIT: <testcase classname="shtest-format.shtest-format" name="unsupported-star.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT: <failure{{[ ]*}}>
# XUNIT: </failure>
# XUNIT-NEXT: </testcase>

# XUNIT: <testcase classname="shtest-format.unsupported_dir" name="some-test.txt" time="{{[0-9]+\.[0-9]+}}">
# XUNIT-NEXT:<skipped message="Skipping because of configuration." />

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
