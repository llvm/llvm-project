; RUN: opt -S -lowertypetests < %s | FileCheck %s

target datalayout = "e-p:32:32"

; Tests that this set of globals is laid out according to our layout algorithm
; (see GlobalLayoutBuilder in include/llvm/Transforms/IPO/LowerTypeTests.h).
; The chosen layout in this case is a, e, b, d, c.

; CHECK: private constant { i32, [0 x i8], i32, [0 x i8], i32, [0 x i8], i32, [0 x i8], i32 } { i32 1, [0 x i8] zeroinitializer, i32 5, [0 x i8] zeroinitializer, i32 2, [0 x i8] zeroinitializer, i32 4, [0 x i8] zeroinitializer, i32 3 }
@a = constant i32 1, !type !0, !type !2
@b = constant i32 2, !type !0, !type !1
@c = constant i32 3, !type !0
@d = constant i32 4, !type !1
@e = constant i32 5, !type !2

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}
!2 = !{i32 0, !"typeid3"}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

define void @foo() {
  %x = call i1 @llvm.type.test(i8* undef, metadata !"typeid1")
  %y = call i1 @llvm.type.test(i8* undef, metadata !"typeid2")
  %z = call i1 @llvm.type.test(i8* undef, metadata !"typeid3")
  ret void
}
