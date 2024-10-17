# Check that lit will not overwrite existing result files when given
# --use-unique-output-file-name.

# Files are overwritten without the option.
# RUN: rm -rf %t.xunit*.xml
# RUN: echo "test" > %t.xunit.xml
# RUN: not %{lit} --xunit-xml-output %t.xunit.xml %{inputs}/xunit-output
# RUN: FileCheck < %t.xunit.xml %s --check-prefix=NEW

# RUN: rm -rf %t.xunit*.xml
# RUN: echo "test" > %t.xunit.xml
# Files should not be overwritten with the option.
# RUN: not %{lit} --xunit-xml-output %t.xunit.xml --use-unique-output-file-name %{inputs}/xunit-output
# RUN: FileCheck < %t.xunit.xml %s --check-prefix=EXISTING
# EXISTING: test
# Results in a new file with "1" added.
# RUN: FileCheck < %t.xunit.1.xml %s --check-prefix=NEW
# NEW:      <?xml version="1.0" encoding="UTF-8"?>
# NEW-NEXT: <testsuites time="{{[0-9.]+}}">
# (assuming that other tests check the whole contents of the file)

# The number should increment as many times as needed.
# RUN: touch %t.xunit.2.xml
# RUN: touch %t.xunit.3.xml
# RUN: touch %t.xunit.4.xml

# RUN: not %{lit} --xunit-xml-output %t.xunit.xml --use-unique-output-file-name %{inputs}/xunit-output
# RUN: FileCheck < %t.xunit.5.xml %s --check-prefix=NEW