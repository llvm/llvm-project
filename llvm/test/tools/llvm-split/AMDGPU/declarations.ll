; RUN: rm -rf %t0 %t1
; RUN: llvm-split -o %t %s -j 2 -mtriple amdgcn-amd-amdhsa
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: not llvm-dis -o - %t1

; Empty module without any defs should result in a single output module that is
; an exact copy of the input.

; CHECK0: declare void @A
; CHECK0: declare void @B

declare void @A()
declare void @B()
