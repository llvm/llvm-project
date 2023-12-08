; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility -pass-remarks=wholeprogramdevirt -stats %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: remark: {{.*}} unique-ret-val: devirtualized a call to vf0
; CHECK: remark: {{.*}} unique-ret-val: devirtualized a call to vf0
; CHECK: remark: {{.*}} devirtualized vf0
; CHECK: remark: {{.*}} devirtualized vf1

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

; CHECK: define i1 @call1
define i1 @call1(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid1")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; CHECK: [[RES1:%[^ ]*]] = icmp eq ptr %vtable, @vt3
  %result = call i1 %fptr(ptr %obj)
  ; CHECK: ret i1 [[RES1]]
  ret i1 %result
}

; CHECK: define i32 @call2
define i32 @call2(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid2")
  call void @llvm.assume(i1 %p)
  %fptr = load ptr, ptr %vtable
  ; Intentional type mismatch to test zero extend.
  ; CHECK: [[RES2:%[^ ]*]] = icmp ne ptr %vtable, @vt2
  %result = call i32 %fptr(ptr %obj)
  ; CHECK: [[ZEXT2:%[^ ]*]] = zext i1 [[RES2]] to i32
  ; CHECK: ret i32 [[ZEXT2:%[^ ]*]]
  ret i32 %result
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}

; CHECK: 2 wholeprogramdevirt - Number of whole program devirtualization targets
; CHECK: 2 wholeprogramdevirt - Number of unique return value optimizations
