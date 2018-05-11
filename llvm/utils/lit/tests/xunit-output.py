# Check xunit output
# RUN: %{lit} --xunit-xml-output %t.xunit.xml %{inputs}/xunit-output || true
# RUN: FileCheck < %t.xunit.xml %s

# CHECK: <?xml version="1.0" encoding="UTF-8" ?>
# CHECK: <testsuites>
# CHECK: <testsuite name='test-data' tests='1' failures='1' skipped='0'>
# CHECK: <testcase classname='test-data.test-data' name='bad&amp;name.ini' time='{{[0-1]}}.{{[0-9]+}}'>
# CHECK-NEXT: 	<failure >
# CHECK-NEXT:&amp; &lt; &gt; "
# CHECK-NEXT:</failure>
# CHECK: </testsuite>
# CHECK: </testsuites>
