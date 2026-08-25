; REQUIRES: asserts
; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility -pass-remarks=wholeprogramdevirt %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: remark: {{.*}} uniform-ret-val: devirtualized a call to vf1

@vt1 = constant [1 x ptr] [ptr @vf1_alias], !type !0
@vt2 = constant [1 x ptr] [ptr @vf2], !type !0

@vf1_alias = alias i32 (ptr), ptr @vf1

define i32 @vf1(ptr %this) readnone {
  ret i32 123
}

define i32 @vf2(ptr %this) readnone {
  ret i32 123
}

; CHECK: define i32 @call
define i32 @call(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  %result = call i32 %fptr(ptr %obj)
  ; CHECK-NOT: call i32 %
  ; CHECK: ret i32 123
  ret i32 %result
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid"}
