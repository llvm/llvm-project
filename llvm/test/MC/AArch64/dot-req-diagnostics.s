// RUN: not llvm-mc -triple aarch64-none-linux-gnu < %s 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-ERROR %s

bar:
        fred .req x5
        fred .req x6
// CHECK-ERROR: warning: ignoring redefinition of register alias 'fred'
// CHECK-ERROR: fred .req x6
// CHECK-ERROR: ^

        ada  .req v2.8b
// CHECK-ERROR: error: vector register without type specifier expected
// CHECK-ERROR: ada  .req v2.8b
// CHECK-ERROR:           ^

        bob  .req lisa
// CHECK-ERROR: error: register name or alias expected
// CHECK-ERROR: bob  .req lisa
// CHECK-ERROR:           ^

        lisa .req x1, 23
// CHECK-ERROR: error: expected newline
// CHECK-ERROR: lisa .req x1, 23
// CHECK-ERROR:             ^

        mov  bob, fred
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: mov  bob, fred
// CHECK-ERROR:      ^

        .unreq 1
// CHECK-ERROR: error: unexpected input in .unreq directive.
// CHECK-ERROR: .unreq 1
// CHECK-ERROR:        ^

        mov  x1, fred
// CHECK: mov x1, x5
// CHECK-NOT: mov x1, x6
