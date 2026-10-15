; Test that a local vtable without uses is not considered a call target,
; while an unused vtable that is not local still is.
; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility -pass-remarks=wholeprogramdevirt %s 2>&1 | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: remark: <unknown>:0:0: single-impl: devirtualized a call to vf1
; CHECK: remark: <unknown>:0:0: single-impl: devirtualized a call to vf3
; CHECK: remark: <unknown>:0:0: devirtualized vf1
; CHECK: remark: <unknown>:0:0: devirtualized vf3
; CHECK-NOT: devirtualized

; There are two vtables with typeid !0. One of the tables is dead, so only one
; remains with typeid !0, meaning that calls through it can be devirtualized.
@vt1 = internal constant [1 x ptr] [ptr @vf1], !type !0
@vt1.dead = internal constant [1 x ptr] [ptr @vf1.dead], !type !0

; There are two vtables with typeid !1. One of the tables is unused, but not
; marked internal, so there might be uses elsewhere. A call with typeid !1
; cannot be devirtualized.
@vt2 = internal constant [1 x ptr] [ptr @vf2], !type !1
@vt2.unused = constant [1 x ptr] [ptr @vf2.unused], !type !1

; A rustc-style vtable: a header of plain data, followed by the methods.
@vt3 = internal constant <{ [8 x i8], ptr }> <{ [8 x i8] zeroinitializer, ptr @vf3 }>, !type !2
@vt3.dead = internal constant [8 x i8] zeroinitializer, !type !2

define void @vf1(ptr %this) {
  ret void
}

define void @vf1.dead(ptr %this) {
  ret void
}

define void @vf2(ptr %this) {
  ret void
}

define void @vf2.unused(ptr %this) {
  ret void
}

define void @vf3(ptr %this) {
  ret void
}

define void @use(ptr %obj1, ptr %obj2, ptr %obj3) {
  ; Use the vt1, vt2 and vt3 vtables so they are not dead.
  store ptr @vt1, ptr %obj1
  store ptr @vt2, ptr %obj2
  store ptr @vt3, ptr %obj3
  ret void
}

; CHECK-LABEL: define void @call1
define void @call1(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 0, metadata !"typeid1")
  %fptr = extractvalue {ptr, i1} %pair, 0
  ; There is only one choice of implementation: the call is devirtualized.
  ; CHECK: call void @vf1(
  call void %fptr(ptr %obj)
  ret void
}

; CHECK-LABEL: define void @call2
define void @call2(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 0, metadata !"typeid2")
  %fptr = extractvalue {ptr, i1} %pair, 0
  ; The non-internal vt2.unused may have uses elsewhere that could end up reaching
  ; this function: the call must remain virtual.
  ; CHECK: call void %{{.*}}(ptr %obj)
  call void %fptr(ptr %obj)
  ret void
}

; CHECK-LABEL: define void @call3
define void @call3(ptr %obj) {
  %vtable = load ptr, ptr %obj
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 8, metadata !"typeid3")
  %fptr = extractvalue {ptr, i1} %pair, 0
  ; There is only one choice of implementation: the call is devirtualized.
  ; CHECK: call void @vf3(
  call void %fptr(ptr %obj)
  ret void
}

declare {ptr, i1} @llvm.type.checked.load(ptr, i32, metadata)

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}
!2 = !{i32 0, !"typeid3"}
