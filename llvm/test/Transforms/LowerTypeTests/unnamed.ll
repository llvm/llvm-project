; RUN: opt -S -lowertypetests < %s | FileCheck %s

target datalayout = "e-p:32:32"

; CHECK: @{{[0-9]+}} = alias
; CHECK: @{{[0-9]+}} = alias
@0 = constant i32 1, !type !0
@1 = constant [2 x i32] [i32 2, i32 3], !type !1

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 4, !"typeid1"}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

define i1 @foo(i8* %p) {
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid1")
  ret i1 %x
}
