; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -b -n 0 -o - %t | llvm-dis | FileCheck --check-prefix=M0 %s
; RUN: llvm-modextract -b -n 1 -o - %t | llvm-dis | FileCheck --check-prefix=M1 %s

; M0: @g = external constant
; M1: @g = constant
@g = constant i8* bitcast (i8** @g to i8*), !type !0

!0 = !{i32 0, !"typeid"}
