; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; Tests bug: https://llvm.org/bugs/show_bug.cgi?id=24645
; CHECK: error: unknown formatter: asm

     !3=!    {ptr asm" !6!={!H)4" ,""  
