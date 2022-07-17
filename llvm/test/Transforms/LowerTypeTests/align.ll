; RUN: opt -S -passes=lowertypetests %s | FileCheck %s

target datalayout = "e-p:32:32"

; CHECK: private constant { i32, [4 x i8], i32 } { i32 1, [4 x i8] zeroinitializer, i32 2 }, align 8
@a = constant i32 1, !type !0
@b = constant i32 2, align 8, !type !0

!0 = !{i32 0, !"typeid1"}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

define i1 @foo(i8* %p) {
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid1")
  ret i1 %x
}
