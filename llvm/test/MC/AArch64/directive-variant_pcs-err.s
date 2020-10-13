// RUN: not llvm-mc -triple aarch64-unknown-none-eabi -filetype asm -o - %s 2>&1 | FileCheck %s

.variant_pcs
// CHECK: error: expected symbol name
// CHECK-NEXT:   .variant_pcs
// CHECK-NEXT:               ^

.variant_pcs foo
// CHECK: error: unknown symbol in '.variant_pcs' directive
// CHECK-NEXT:   .variant_pcs foo
// CHECK-NEXT:                ^

.global foo
.variant_pcs foo bar
// CHECK: error: unexpected token in '.variant_pcs' directive
// CHECK-NEXT:   .variant_pcs foo bar
// CHECK-NEXT:                    ^
