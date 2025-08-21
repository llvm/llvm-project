# RUN: %{lit} --ignore-fail --show-pass %{inputs}/print-relative-path | FileCheck --check-prefix=CHECK-DEFAULT %s
# RUN: %{lit} --ignore-fail --show-pass -r %{inputs}/print-relative-path | FileCheck --check-prefix=CHECK-RELATIVE %s
# RUN: %{lit} --ignore-fail --show-pass --relative-paths %{inputs}/print-relative-path | FileCheck --check-prefix=CHECK-RELATIVE %s


# CHECK-DEFAULT: PASS: print-relative-path :: test.txt (1 of 2)
# CHECK-DEFAULT: FAIL: print-relative-path :: test2.txt (2 of 2)
# CHECK-DEFAULT: Passed Tests (1):
# CHECK-DEFAULT:  print-relative-path :: test.txt
# CHECK-DEFAULT: Failed Tests (1):
# CHECK-DEFAULT:  print-relative-path :: test2.txt

# CHECK-RELATIVE: PASS: print-relative-path :: test.txt (1 of 2)
# CHECK-RELATIVE: FAIL: print-relative-path :: test2.txt (2 of 2)
# CHECK-RELATIVE: Passed Tests (1):
# CHECK-RELATIVE:  Inputs[/\\]print-relative-path[/\\]test.txt
# CHECK-RELATIVE: Failed Tests (1):
# CHECK-RELATIVE:  Inputs[/\\]print-relative-path[/\\]test2.txt
