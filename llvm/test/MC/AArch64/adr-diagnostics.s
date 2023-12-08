// RUN: not llvm-mc -triple aarch64-none-linux-gnu %s -filetype=obj -o /dev/null 2>&1 | FileCheck %s

  adr x0, -start
// CHECK: error: expected relocatable expression
// CHECK-NEXT:   adr x0, -start
// CHECK-NEXT:   ^
  adr x1, start * 10
// CHECK: error: expected relocatable expression
// CHECK-NEXT:   adr x1, start * 10
// CHECK-NEXT:   ^
  adr x2, 2 * (start + 987136)
// CHECK: error: expected relocatable expression
// CHECK-NEXT:   adr x2, 2 * (start + 987136)
// CHECK-NEXT:   ^
  adr x3, (end + start)
// CHECK: error: expected relocatable expression
// CHECK-NEXT:   adr x3, (end + start)
// CHECK-NEXT:   ^
  adr x4, #(end - start)
// CHECK: error: symbol 'start' can not be undefined in a subtraction expression
// CHECK-NEXT:   adr x4, #(end - start)
// CHECK-NEXT:   ^

  adrp x0, -start
// CHECK: error: expected relocatable expression
// CHECK-NEXT:   adrp x0, -start
// CHECK-NEXT:   ^
  adrp x1, start * 10
// CHECK: error: expected relocatable expression
// CHECK-NEXT:   adrp x1, start * 10
// CHECK-NEXT:   ^
  adrp x2, 2 * (start + 987136)
// CHECK: error: expected relocatable expression
// CHECK-NEXT:   adrp x2, 2 * (start + 987136)
// CHECK-NEXT:   ^
  adrp x3, (end + start)
// CHECK: error: expected relocatable expression
// CHECK-NEXT:   adrp x3, (end + start)
// CHECK-NEXT:   ^
  adrp x4, #(end - start)
// CHECK: error: symbol 'start' can not be undefined in a subtraction expression
// CHECK-NEXT:   adrp x4, #(end - start)
// CHECK-NEXT:   ^
