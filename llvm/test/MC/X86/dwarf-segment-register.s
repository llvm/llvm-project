// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t.64
// RUN: llvm-objdump --dwarf=frames %t.64 | FileCheck %s --check-prefixes=X64,CHECK
// RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -o %t.32
// RUN: llvm-objdump --dwarf=frames %t.32 | FileCheck %s --check-prefixes=X86,CHECK

.cfi_startproc
.cfi_offset %cs, -40
.cfi_offset %ds, -32
.cfi_offset %ss, -24
.cfi_offset %es, -16
.cfi_offset %fs, -8
.cfi_offset %gs, 0
.cfi_endproc

// X64: reg51
// X86: reg41
// CHECK-SAME: -40

// X64: reg53
// X86: reg43
// CHECK-SAME: -32

// X64: reg52
// X86: reg42
// CHECK-SAME: -24

// X64: reg50
// X86: reg40
// CHECK-SAME: -16

// X64: reg54
// X86: reg44
// CHECK-SAME: -8

// X64: reg55
// X86: reg45
// CHECK-SAME: 0
