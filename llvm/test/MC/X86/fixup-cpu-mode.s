// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t
// RUN: llvm-objdump --no-print-imm-hex -d %t | FileCheck %s

//PR18303
.global edata
sub $edata, %r12 // CHECK: subq $0, %r12
.code32

