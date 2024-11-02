; RUN: rm -rf %t0 %t1
; RUN: llvm-split -o %t %s -j 2 -mtriple amdgcn-amd-amdhsa
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; Check that all declarations are put into each partition.

; CHECK0: declare void @A
; CHECK0: declare void @B

; CHECK1: declare void @A
; CHECK1: declare void @B

declare void @A()

declare void @B()
