// RUN: llvm-mc -triple=arm64-apple-ios -filetype=obj %s -o %t
// RUN: llvm-objdump --macho --unwind-info --dwarf=frames %t | FileCheck %s

// Check that we fallback on DWARF instead of asserting.

// CHECK: Contents of __compact_unwind section:
// CHECK: compact encoding:     0x03000000
// CHECK: compact encoding:     0x03000000

// CHECK: .eh_frame contents:
// CHECK: DW_CFA_def_cfa: reg1 +32

//  DW_CFA_def_cfa_offset: +32
//  DW_CFA_def_cfa_offset: +64

_cfi_dwarf0:
 .cfi_startproc
 .cfi_def_cfa x1, 32;
 .cfi_endproc

_cfi_dwarf1:
 .cfi_startproc
 .cfi_def_cfa_offset 32
 .cfi_def_cfa_offset 64
 .cfi_endproc
