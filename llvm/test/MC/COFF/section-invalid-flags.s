// RUN: not llvm-mc -triple i386-pc-win32 -filetype=obj %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple x86_64-pc-win32 -filetype=obj %s 2>&1 | FileCheck %s

// CHECK: error: conflicting section flags 'b' and 'd'
.section s_db,"db"; .long 1

// CHECK: error: conflicting section flags 'b' and 'd'
.section s_bd,"bd"; .long 1

// CHECK: error: expected comdat type such as 'discard' or 'largest' after protection bits
.section .stack, "w", @nobits
