// RUN: not llvm-mc -filetype=obj -triple i386-pc-win32 %s 2>&1 | FileCheck %s
// CHECK: assembler label 'Lundefined' can not be undefined
// CHECK: assembler label 'Lundefined2' can not be undefined
        .text
        movl Lundefined, %eax
        movl Lundefined2, %eax
