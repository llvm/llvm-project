; RUN: opt -S -lowertypetests < %s | FileCheck %s

target datalayout = "e-p:32:32"

; CHECK: [[G:@[^ ]*]] = private constant { i32, [0 x i8], i32 }
@a = constant i32 1, !type !0, !type !1
@b = constant i32 2, !type !0, !type !2

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}
!2 = !{i32 0, !"typeid3"}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

; CHECK: @foo(i8* [[A0:%[^ ]*]])
define i1 @foo(i8* %p) {
  ; CHECK: [[R0:%[^ ]*]] = ptrtoint i8* [[A0]] to i32
  ; CHECK: [[R1:%[^ ]*]] = icmp eq i32 [[R0]], ptrtoint ({ i32, [0 x i8], i32 }* [[G]] to i32)
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid2")
  ; CHECK: ret i1 [[R1]]
  ret i1 %x
}

; CHECK: @bar(i8* [[B0:%[^ ]*]])
define i1 @bar(i8* %p) {
  ; CHECK: [[S0:%[^ ]*]] = ptrtoint i8* [[B0]] to i32
  ; CHECK: [[S1:%[^ ]*]] = icmp eq i32 [[S0]], add (i32 ptrtoint ({ i32, [0 x i8], i32 }* [[G]] to i32), i32 4)
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid3")
  ; CHECK: ret i1 [[S1]]
  ret i1 %x
}

; CHECK: @x(
define i1 @x(i8* %p) {
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid1")
  ret i1 %x
}
