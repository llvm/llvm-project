## Check xunit output.
# RUN: not %{lit} --report-failures-only --xunit-xml-output %t.xunit.xml %{inputs}/xunit-output
# RUN: FileCheck --input-file=%t.xunit.xml %s

# CHECK:      <?xml version="1.0" encoding="UTF-8"?>
# CHECK-NEXT: <testsuites time="{{[0-9.]+}}">
# CHECK-NEXT: <testsuite name="test-data" tests="1" failures="1" skipped="0" time="{{[0-9.]+}}">
# CHECK-NEXT: <testcase classname="test-data.test-data" name="bad&amp;name.ini" time="{{[0-9.]+}}">
# CHECK-NEXT:   <failure><![CDATA[& < > ]]]]><![CDATA[> &"]]></failure>
# CHECK-NEXT: </testcase>
# CHECK-NEXT: </testsuite>
# CHECK-NEXT: </testsuites>
