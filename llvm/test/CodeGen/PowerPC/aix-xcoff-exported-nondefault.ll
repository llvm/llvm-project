; RUN: not llc -filetype=null -mtriple powerpc-ibm-aix-xcoff 2>&1 %s | FileCheck %s
; RUN: not llc -filetype=null -mtriple powerpc64-ibm-aix-xcoff 2>&1 %s | FileCheck %s

; CHECK: dllexport GlobalValue must have default visibility
@b_e = hidden dllexport global i32 0, align 4
