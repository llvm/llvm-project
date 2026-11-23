// RUN: llvm-mc -triple aarch64-elf -filetype=obj %s -o - | llvm-objdump -r - | FileCheck %s

// Test that PATCHINST appears after JUMP26.
// CHECK:      R_AARCH64_JUMP26
// CHECK-NEXT: R_AARCH64_JUMP26
// CHECK-NEXT: R_AARCH64_PATCHINST
// CHECK-NEXT: R_AARCH64_PATCHINST
.reloc ., R_AARCH64_PATCHINST, ds
b f1
.balign 8
.reloc ., R_AARCH64_PATCHINST, ds
b f2
