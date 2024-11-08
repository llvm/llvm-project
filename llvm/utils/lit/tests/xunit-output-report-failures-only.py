# UNSUPPORTED: system-windows

## Check xunit output.
# RUN: rm -rf %t.xunit.xml
# RUN: not %{lit} --report-failures-only --xunit-xml-output %t.xunit.xml %{inputs}/xunit-output
# If xmllint is installed verify that the generated xml is well-formed
# RUN: sh -c 'if command -v xmllint 2>/dev/null; then xmllint --noout %t.xunit.xml; fi'
# RUN: FileCheck < %t.xunit.xml %s

# CHECK:      <?xml version="1.0" encoding="UTF-8"?>
# CHECK-NEXT: <testsuites time="{{[0-9.]+}}">
# CHECK-NEXT: <testsuite name="test-data" tests="5" failures="1" skipped="3" time="{{[0-9.]+}}">
# CHECK-NEXT: <testcase classname="test-data.test-data" name="bad&amp;name.ini" time="{{[0-1]\.[0-9]+}}">
# CHECK-NEXT:   <failure><![CDATA[& < > ]]]]><![CDATA[> &"]]></failure>
# CHECK-NEXT: </testcase>
# CHECK-NEXT: </testsuite>
# CHECK-NEXT: </testsuites>
