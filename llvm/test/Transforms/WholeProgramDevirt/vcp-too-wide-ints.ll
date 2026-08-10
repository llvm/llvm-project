; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@vt1 = constant [1 x ptr] [ptr @vf1], !type !0
@vt2 = constant [1 x ptr] [ptr @vf2], !type !0
@vt3 = constant [1 x ptr] [ptr @vf3], !type !1
@vt4 = constant [1 x ptr] [ptr @vf4], !type !1

define i64 @vf1(ptr %this, i128 %arg) readnone {
  %argtrunc = trunc i128 %arg to i64
  ret i64 %argtrunc
}

define i64 @vf2(ptr %this, i128 %arg) readnone {
  %argtrunc = trunc i128 %arg to i64
  ret i64 %argtrunc
}

define i128 @vf3(ptr %this, i64 %arg) readnone {
  %argzext = zext i64 %arg to i128
  ret i128 %argzext
}

define i128 @vf4(ptr %this, i64 %arg) readnone {
  %argzext = zext i64 %arg to i128
  ret i128 %argzext
}

; CHECK: define i64 @call1
define i64 @call1(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; CHECK: call i64 %
  %result = call i64 %fptr(ptr %obj, i128 1)
  ret i64 %result
}

; CHECK: define i128 @call2
define i128 @call2(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; CHECK: call i128 %
  %result = call i128 %fptr(ptr %obj, i64 1)
  ret i128 %result
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}
