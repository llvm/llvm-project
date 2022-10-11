; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant [1 x ptr] [ptr @vf0], !type !0
@vt2 = constant [1 x ptr] [ptr @vf0], !type !0, !type !1
@vt3 = constant [1 x ptr] [ptr @vf1], !type !0, !type !1
@vt4 = constant [1 x ptr] [ptr @vf1], !type !1

define i1 @vf0(ptr %this) readnone {
  ret i1 0
}

define i1 @vf1(ptr %this) readnone {
  ret i1 1
}

; CHECK: define i1 @call
define i1 @call(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %p2 = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid1")
  call void @llvm.assume(i1 %p2)
  %fptr = load ptr, ptr %vtable
  ; CHECK: [[RES1:%[^ ]*]] = icmp eq ptr %vtable, @vt3
  %result = call i1 %fptr(ptr %obj)
  ; CHECK: ret i1 [[RES1]]
  ret i1 %result
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}
