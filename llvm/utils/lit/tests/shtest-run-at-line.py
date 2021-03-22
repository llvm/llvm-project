# Check that -vv makes the line number of the failing RUN command clear.
# (-v is actually sufficient in the case of the internal shell.)

# FIXME: this test depends on order of tests
# RUN: rm -f %{inputs}/shtest-run-at-line/.lit_test_times.txt

# RUN: not %{lit} -j 1 -vv %{inputs}/shtest-run-at-line > %t.out
# RUN: FileCheck --input-file %t.out %s
#
# END.


# CHECK: Testing: 4 tests


# In the case of the external shell, we check for only RUN lines in stderr in
# case some shell implementations format "set -x" output differently.

# CHECK-LABEL: FAIL: shtest-run-at-line :: external-shell/basic.txt

# CHECK:      Script:
# CHECK:      RUN: at line 4{{.*}}  true
# CHECK-NEXT: RUN: at line 5{{.*}}  false
# CHECK-NEXT: RUN: at line 6{{.*}}  true

# CHECK:     RUN: at line 4
# CHECK:     RUN: at line 5
# CHECK-NOT: RUN

# CHECK-LABEL: FAIL: shtest-run-at-line :: external-shell/line-continuation.txt

# CHECK:      Script:
# CHECK:      RUN: at line 4{{.*}}  echo 'foo bar'  | FileCheck
# CHECK-NEXT: RUN: at line 6{{.*}}  echo 'foo baz'  | FileCheck
# CHECK-NEXT: RUN: at line 9{{.*}}  echo 'foo bar'  | FileCheck

# CHECK:     RUN: at line 4
# CHECK:     RUN: at line 6
# CHECK-NOT: RUN


# CHECK-LABEL: FAIL: shtest-run-at-line :: internal-shell/basic.txt

# CHECK:      Script:
# CHECK:      : 'RUN: at line 1';  true
# CHECK-NEXT: : 'RUN: at line 2';  false
# CHECK-NEXT: : 'RUN: at line 3';  true

# CHECK:      Command Output (stdout)
# CHECK:      $ ":" "RUN: at line 1"
# CHECK-NEXT: $ "true"
# CHECK-NEXT: $ ":" "RUN: at line 2"
# CHECK-NEXT: $ "false"
# CHECK-NOT:  RUN

# CHECK-LABEL: FAIL: shtest-run-at-line :: internal-shell/line-continuation.txt

# CHECK:      Script:
# CHECK:      : 'RUN: at line 1';  : first line continued to second line
# CHECK-NEXT: : 'RUN: at line 3';  echo 'foo bar'  | FileCheck
# CHECK-NEXT: : 'RUN: at line 5';  echo  'foo baz'  | FileCheck
# CHECK-NEXT: : 'RUN: at line 8';  echo 'foo bar'  | FileCheck

# CHECK:      Command Output (stdout)
# CHECK:      $ ":" "RUN: at line 1"
# CHECK-NEXT: $ ":" "first" "line" "continued" "to" "second" "line"
# CHECK-NEXT: $ ":" "RUN: at line 3"
# CHECK-NEXT: $ "echo" "foo bar"
# CHECK-NEXT: $ "FileCheck" "{{.*}}"
# CHECK-NEXT: $ ":" "RUN: at line 5"
# CHECK-NEXT: $ "echo" "foo baz"
# CHECK-NEXT: $ "FileCheck" "{{.*}}"
# CHECK-NOT:  RUN
