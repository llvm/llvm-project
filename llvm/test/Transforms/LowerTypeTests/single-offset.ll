; RUN: opt -S -passes=lowertypetests %s | FileCheck %s

target datalayout = "e-p:32:32"

; CHECK: [[G:@[^ ]*]] = private constant { i32, [0 x i8], i32 }
@a = constant i32 1, !type !0, !type !1
@b = constant i32 2, !type !0, !type !2

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}
!2 = !{i32 0, !"typeid3"}

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset) nounwind readnone

; CHECK: @foo(ptr [[A0:%[^ ]*]])
define i1 @foo(ptr %p) {
  ; CHECK: [[R0:%[^ ]*]] = ptrtoint ptr [[A0]] to i32
  ; CHECK: [[R1:%[^ ]*]] = icmp eq i32 [[R0]], ptrtoint (ptr [[G]] to i32)
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid2")
  ; CHECK: ret i1 [[R1]]
  ret i1 %x
}

; CHECK: @bar(ptr [[B0:%[^ ]*]])
define i1 @bar(ptr %p) {
  ; CHECK: [[S0:%[^ ]*]] = ptrtoint ptr [[B0]] to i32
  ; CHECK: [[S1:%[^ ]*]] = icmp eq i32 [[S0]],  ptrtoint (ptr getelementptr (i8, ptr [[G]], i32 4) to i32)
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid3")
  ; CHECK: ret i1 [[S1]]
  ret i1 %x
}

; CHECK: @x(
define i1 @x(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid1")
  ret i1 %x
}
