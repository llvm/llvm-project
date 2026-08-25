# Check that -a/-v/-vv makes the line number of the failing RUN command clear.


# RUN: not %{lit} -a %{inputs}/shtest-run-at-line | %{filter-lit} | FileCheck %s
# RUN: not %{lit} -v %{inputs}/shtest-run-at-line | %{filter-lit} | FileCheck %s
# RUN: not %{lit} -vv %{inputs}/shtest-run-at-line | %{filter-lit} | FileCheck %s
# END.


# CHECK: Testing: 4 tests


# CHECK-LABEL: FAIL: shtest-run-at-line :: basic.txt

# CHECK:      Command Output (stdout)
# CHECK-NEXT: --
# CHECK-NEXT: # RUN: at line 1
# CHECK-NEXT: true
# CHECK-NEXT: # executed command: true
# CHECK-NEXT: # RUN: at line 2
# CHECK-NEXT: false
# CHECK-NEXT: # executed command: false
# CHECK-NOT:  RUN

# CHECK-LABEL: FAIL: shtest-run-at-line :: empty-run-line.txt

#      CHECK: Command Output (stdout)
# CHECK-NEXT: --
# CHECK-NEXT: # RUN: at line 2 has no command after substitutions
# CHECK-NEXT: # RUN: at line 3
# CHECK-NEXT: false
# CHECK-NEXT: # executed command: false
#  CHECK-NOT: RUN

# CHECK-LABEL: FAIL: shtest-run-at-line :: line-continuation.txt

# CHECK:      Command Output (stdout)
# CHECK-NEXT: --
# CHECK-NEXT: # RUN: at line 1
# CHECK-NEXT: : first line continued to second line
# CHECK-NEXT: # executed command: : first line continued to second line
# CHECK-NEXT: # RUN: at line 3
# CHECK-NEXT: echo 'foo bar' | FileCheck {{.*}}
# CHECK-NEXT: # executed command: echo 'foo bar'
# CHECK-NEXT: # executed command: FileCheck {{.*}}
# CHECK-NEXT: # RUN: at line 5
# CHECK-NEXT: echo 'foo baz' | FileCheck {{.*}}
# CHECK-NEXT: # executed command: echo 'foo baz'
# CHECK-NEXT: # executed command: FileCheck {{.*}}
# CHECK-NOT:  RUN

# CHECK-LABEL: FAIL: shtest-run-at-line :: run-line-with-newline.txt

#      CHECK: Command Output (stdout)
# CHECK-NEXT: --
# CHECK-NEXT: # RUN: at line 1
# CHECK-NEXT: echo abc |
# CHECK-NEXT: FileCheck {{.*}} &&
# CHECK-NEXT: false
# CHECK-NEXT: # executed command: echo abc
# CHECK-NEXT: # executed command: FileCheck {{.*}}
# CHECK-NEXT: # executed command: false
#  CHECK-NOT: RUN
