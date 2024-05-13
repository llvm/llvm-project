// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t.64
// RUN: llvm-objdump --dwarf=frames %t.64 | FileCheck %s --check-prefixes=64,CHECK
// RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -o %t.32
// RUN: llvm-objdump --dwarf=frames %t.32 | FileCheck %s --check-prefixes=32,CHECK

.cfi_startproc
.cfi_offset %cs, -40
.cfi_offset %ds, -32
.cfi_offset %ss, -24
.cfi_offset %es, -16
.cfi_offset %fs, -8
.cfi_offset %gs, 0
.cfi_endproc

// 64: reg51
// 32: reg41
// CHECK-SAME: -40

// 64: reg53
// 32: reg43
// CHECK-SAME: -32

// 64: reg52
// 32: reg42
// CHECK-SAME: -24

// 64: reg50
// 32: reg40
// CHECK-SAME: -16

// 64: reg54
// 32: reg44
// CHECK-SAME: -8

// 64: reg55
// 32: reg45
// CHECK-SAME: 0
