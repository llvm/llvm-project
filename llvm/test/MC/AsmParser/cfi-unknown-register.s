// RUN: llvm-mc -filetype=asm -triple x86_64-pc-linux-gnu %s 2>&1 | FileCheck %s

.cfi_sections .debug_frame
.cfi_startproc
.cfi_rel_offset 99, 0
// CHECK: .cfi_rel_offset 99, 0
.cfi_endproc
