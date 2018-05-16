# Check xunit output
# RUN: rm -rf %t.xunit.xml
# RUN: not %{lit} --xunit-xml-output %t.xunit.xml %{inputs}/xunit-output
# If xmllint is installed verify that the generated xml is well-formed
# RUN: sh -c 'if command -v xmllint 2>/dev/null; then xmllint --noout %t.xunit.xml; fi'
# RUN: FileCheck < %t.xunit.xml %s

# CHECK: <?xml version="1.0" encoding="UTF-8" ?>
# CHECK: <testsuites>
# CHECK: <testsuite name='test-data' tests='1' failures='1' skipped='0'>
# CHECK: <testcase classname='test-data.test-data' name='bad&amp;name.ini' time='{{[0-1]}}.{{[0-9]+}}'>
# CHECK-NEXT: <failure ><![CDATA[& < > ]]]]><![CDATA[> &"]]></failure>
# CHECK: </testsuite>
# CHECK: </testsuites>
