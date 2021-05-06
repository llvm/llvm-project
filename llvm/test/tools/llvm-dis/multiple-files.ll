; RUN: llvm-as -o %t0 %s
; RUN: cp %t0 %t1
; RUN: not llvm-dis -o %t2 %t0 %t1 2>&1 | FileCheck %s --check-prefix ERROR
; RUN: llvm-dis %t0 %t1
; RUN: FileCheck %s < %t0.ll
; RUN: FileCheck %s < %t1.ll
; ERROR: error: output file name cannot be set for multiple input files

; CHECK: declare void @foo
declare void @foo() 
