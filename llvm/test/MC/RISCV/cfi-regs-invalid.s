# RUN: not llvm-mc -triple riscv32 < %s 2>&1 | FileCheck %s
# RUN: not llvm-mc -triple riscv64 < %s 2>&1 | FileCheck %s

.cfi_startproc
.cfi_offset x00, 0 # CHECK: :[[@LINE]]:16: error: invalid register name
.cfi_offset a8, 8 # CHECK: :[[@LINE]]:15: error: invalid register name
.cfi_endproc
