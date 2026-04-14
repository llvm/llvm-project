// RUN: not llvm-mc -triple aarch64_lfi %s 2>&1 | FileCheck %s

mov x27, x0
// CHECK: error: illegal modification of reserved LFI register
// CHECK:        mov x27, x0

ldr x27, [x0]
// CHECK: error: illegal modification of reserved LFI register
// CHECK:        ldr x27, [x0]

add x27, x0, x1
// CHECK: error: illegal modification of reserved LFI register
// CHECK:        add x27, x0, x1

mov x28, x0
// CHECK: error: illegal modification of reserved LFI register
// CHECK:        mov x28, x0

ldr x28, [x0]
// CHECK: error: illegal modification of reserved LFI register
// CHECK:        ldr x28, [x0]

add x28, x0, x1
// CHECK: error: illegal modification of reserved LFI register
// CHECK:        add x28, x0, x1

ldp x27, x28, [x0]
// CHECK: error: illegal modification of reserved LFI register
// CHECK:        ldp x27, x28, [x0]

ldp x0, x27, [x1]
// CHECK: error: illegal modification of reserved LFI register
// CHECK:        ldp x0, x27, [x1]

ldp x28, x0, [x1]
// CHECK: error: illegal modification of reserved LFI register
// CHECK:        ldp x28, x0, [x1]

ldr x0, [x27], #8
// CHECK: error: illegal modification of reserved LFI register
// CHECK:        ldr x0, [x27], #8

ldr x0, [x28, #8]!
// CHECK: error: illegal modification of reserved LFI register
// CHECK:        ldr x0, [x28, #8]!
