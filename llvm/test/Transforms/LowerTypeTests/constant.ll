; RUN: opt -S -lowertypetests < %s | FileCheck %s
; RUN: opt -S -passes=lowertypetests < %s | FileCheck %s

target datalayout = "e-p:32:32"

@a = constant i32 1, !type !0
@b = constant [2 x i32] [i32 2, i32 3], !type !1

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 4, !"typeid1"}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

; CHECK: @foo(
define i1 @foo() {
  ; CHECK: ret i1 true
  %x = call i1 @llvm.type.test(i8* bitcast (i32* @a to i8*), metadata !"typeid1")
  ret i1 %x
}

; CHECK: @bar(
define i1 @bar() {
  ; CHECK: ret i1 true
  %x = call i1 @llvm.type.test(i8* bitcast (i32* getelementptr ([2 x i32], [2 x i32]* @b, i32 0, i32 1) to i8*), metadata !"typeid1")
  ret i1 %x
}

; CHECK: @baz(
define i1 @baz() {
  ; CHECK-NOT: ret i1 true
  %x = call i1 @llvm.type.test(i8* bitcast (i32* getelementptr ([2 x i32], [2 x i32]* @b, i32 0, i32 0) to i8*), metadata !"typeid1")
  ret i1 %x
}
