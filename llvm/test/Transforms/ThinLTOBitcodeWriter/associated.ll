; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -b -n 0 -o - %t | llvm-dis | FileCheck --check-prefix=M0 %s
; RUN: llvm-modextract -b -n 1 -o - %t | llvm-dis | FileCheck --check-prefix=M1 %s

; M0: @g = external constant
; M0-NOT: @assoc
; M1: @g = constant i8 1
; M1: @assoc = private constant i8 2

@g = constant i8 1, !type !0
@assoc = private constant i8 2, !associated !1

!0 = !{i32 0, !"typeid"}
!1 = !{ptr @g}
