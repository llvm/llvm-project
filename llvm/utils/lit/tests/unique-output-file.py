## Check that lit will not overwrite existing result files when given
## --use-unique-output-file-name.

## Files are overwritten without the option.
# RUN: rm -f %t.xunit*.xml
# RUN: echo "test" > %t.xunit.xml
# RUN: not %{lit} --xunit-xml-output %t.xunit.xml %{inputs}/xunit-output
# RUN: FileCheck < %t.xunit.xml %s --check-prefix=NEW
# NEW:      <?xml version="1.0" encoding="UTF-8"?>
# NEW-NEXT: <testsuites time="{{[0-9.]+}}">
## (other tests will check the contents of the whole file)

# RUN: rm -f %t.xunit*.xml
# RUN: echo "test" > %t.xunit.xml
## Files should not be overwritten with the option.
# RUN: not %{lit} --xunit-xml-output %t.xunit.xml --use-unique-output-file-name %{inputs}/xunit-output
# RUN: FileCheck < %t.xunit.xml %s --check-prefix=EXISTING
# EXISTING: test
## Results in a new file with some discriminator added.
# RUN: ls -l %t.xunit*.xml | wc -l | FileCheck %s --check-prefix=NUMFILES
# NUMFILES: 2
# RUN: FileCheck < %t.xunit.*.xml %s --check-prefix=NEW
