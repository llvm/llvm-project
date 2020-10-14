// RUN: llvm-mc -triple aarch64-elf -filetype asm -o - %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-elf -filetype obj -o - %s \
// RUN:   | llvm-readobj --symbols - | FileCheck %s --check-prefix=CHECK-ST_OTHER

.text
.global foo
.variant_pcs foo
// CHECK: .variant_pcs foo

// CHECK-ST_OTHER: Name: foo
// CHECK-ST_OTHER: Other [ (0x80)
