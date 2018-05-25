# RUN: rm -f %t.xml
# RUN: not %{lit} -j 1 -v %{inputs}/shtest-format --xunit-xml-output %t.xml
# RUN: FileCheck < %t.xml %s

# CHECK: <?xml version="1.0" encoding="UTF-8" ?>
# CHECK-NEXT: <testsuites>
# CHECK-NEXT: <testsuite name="shtest-format" tests="23" failures="7" skipped="5">

# CHECK: <testcase classname="shtest-format.shtest-format" name="argv0.txt" time="{{[0-9]+\.[0-9]+}}"/>

# CHECK: <testcase classname="shtest-format.external_shell" name="fail.txt" time="{{[0-9]+\.[0-9]+}}">
# CHECK-NEXT: <failure{{[ ]*}}>
# CHECK: </failure>
# CHECK-NEXT: </testcase>


# CHECK: <testcase classname="shtest-format.external_shell" name="fail_with_bad_encoding.txt" time="{{[0-9]+\.[0-9]+}}">
# CHECK-NEXT: <failure{{[ ]*}}>
# CHECK: </failure>
# CHECK-NEXT: </testcase>

# CHECK: <testcase classname="shtest-format.external_shell" name="pass.txt" time="{{[0-9]+\.[0-9]+}}"/>

# CHECK: <testcase classname="shtest-format.shtest-format" name="fail.txt" time="{{[0-9]+\.[0-9]+}}">
# CHECK-NEXT: <failure{{[ ]*}}>
# CHECK: </failure>
# CHECK-NEXT: </testcase>

# CHECK: <testcase classname="shtest-format.shtest-format" name="no-test-line.txt" time="{{[0-9]+\.[0-9]+}}">
# CHECK-NEXT: <failure{{[ ]*}}>
# CHECK: </failure>
# CHECK-NEXT: </testcase>

# CHECK: <testcase classname="shtest-format.shtest-format" name="pass.txt" time="{{[0-9]+\.[0-9]+}}"/>

# CHECK: <testcase classname="shtest-format.shtest-format" name="requires-any-missing.txt" time="{{[0-9]+\.[0-9]+}}">
# CHECK-NEXT:<skipped message="Skipping because of: a-missing-feature || a-missing-feature-2" />

# CHECK: <testcase classname="shtest-format.shtest-format" name="requires-any-present.txt" time="{{[0-9]+\.[0-9]+}}"/>

# CHECK: <testcase classname="shtest-format.shtest-format" name="requires-missing.txt" time="{{[0-9]+\.[0-9]+}}">
# CHECK-NEXT:<skipped message="Skipping because of: a-missing-feature" />

# CHECK: <testcase classname="shtest-format.shtest-format" name="requires-present.txt" time="{{[0-9]+\.[0-9]+}}"/>

# CHECK: <testcase classname="shtest-format.shtest-format" name="requires-star.txt" time="{{[0-9]+\.[0-9]+}}">
# CHECK-NEXT: <failure{{[ ]*}}>
# CHECK: </failure>
# CHECK-NEXT: </testcase>


# CHECK: <testcase classname="shtest-format.shtest-format" name="requires-triple.txt" time="{{[0-9]+\.[0-9]+}}">
# CHECK-NEXT:<skipped message="Skipping because of: x86_64" />

# CHECK: <testcase classname="shtest-format.shtest-format" name="unsupported-expr-false.txt" time="{{[0-9]+\.[0-9]+}}"/>

# CHECK: <testcase classname="shtest-format.shtest-format" name="unsupported-expr-true.txt" time="{{[0-9]+\.[0-9]+}}">
# CHECK-NEXT:<skipped message="Skipping because of configuration." />

# CHECK: <testcase classname="shtest-format.shtest-format" name="unsupported-star.txt" time="{{[0-9]+\.[0-9]+}}">
# CHECK-NEXT: <failure{{[ ]*}}>
# CHECK: </failure>
# CHECK-NEXT: </testcase>

# CHECK: <testcase classname="shtest-format.unsupported_dir" name="some-test.txt" time="{{[0-9]+\.[0-9]+}}">
# CHECK-NEXT:<skipped message="Skipping because of configuration." />

# CHECK: <testcase classname="shtest-format.shtest-format" name="xfail-expr-false.txt" time="{{[0-9]+\.[0-9]+}}"/>

# CHECK: <testcase classname="shtest-format.shtest-format" name="xfail-expr-true.txt" time="{{[0-9]+\.[0-9]+}}"/>

# CHECK: <testcase classname="shtest-format.shtest-format" name="xfail-feature.txt" time="{{[0-9]+\.[0-9]+}}"/>

# CHECK: <testcase classname="shtest-format.shtest-format" name="xfail-target.txt" time="{{[0-9]+\.[0-9]+}}"/>

# CHECK: <testcase classname="shtest-format.shtest-format" name="xfail.txt" time="{{[0-9]+\.[0-9]+}}"/>

# CHECK: <testcase classname="shtest-format.shtest-format" name="xpass.txt" time="{{[0-9]+\.[0-9]+}}">
# CHECK-NEXT: <failure{{[ ]*}}>
# CHECK: </failure>
# CHECK-NEXT: </testcase>

# CHECK: </testsuite>
# CHECK-NEXT: </testsuites>
