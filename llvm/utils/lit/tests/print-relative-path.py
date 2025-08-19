# RUN: not %{lit} %{inputs}/print-relative-path | FileCheck --check-prefix=CHECK-DEFAULT %s
# RUN: not %{lit} -r %{inputs}/print-relative-path | FileCheck --check-prefix=CHECK-RELATIVE %s
# RUN: not %{lit} --relative-paths %{inputs}/print-relative-path | FileCheck --check-prefix=CHECK-RELATIVE %s


# CHECK-DEFAULT: PASS: print-relative-path :: test.txt (1 of 2)
# CHECK-DEFAULT-NEXT: FAIL: print-relative-path :: test2.txt (2 of 2)
# CHECK-DEFAULT-NEXT: ********************
# CHECK-DEFAULT-NEXT: Failed Tests (1):
# CHECK-DEFAULT-NEXT:  print-relative-path :: test2.txt

# CHECK-RELATIVE: PASS: print-relative-path :: Inputs/print-relative-path/test.txt (1 of 2)
# CHECK-RELATIVE-NEXT: FAIL: print-relative-path :: Inputs/print-relative-path/test2.txt (2 of 2)
# CHECK-RELATIVE-NEXT: ********************
# CHECK-RELATIVE-NEXT: Failed Tests (1):
# CHECK-RELATIVE-NEXT:  print-relative-path :: Inputs/print-relative-path/test2.txt
