; RUN: opt < %s -passes=tsan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare void @escape(i32*)

@sink = global i32* null, align 4

define void @captured0() nounwind uwtable sanitize_thread {
entry:
  %ptr = alloca i32, align 4
  ; escapes due to call
  call void @escape(i32* %ptr)
  store i32 42, i32* %ptr, align 4
  ret void
}
; CHECK-LABEL: define void @captured0
; CHECK: __tsan_write
; CHECK: ret void

define void @captured1() nounwind uwtable sanitize_thread {
entry:
  %ptr = alloca i32, align 4
  ; escapes due to store into global
  store i32* %ptr, i32** @sink, align 8
  store i32 42, i32* %ptr, align 4
  ret void
}
; CHECK-LABEL: define void @captured1
; CHECK: __tsan_write
; CHECK: __tsan_write
; CHECK: ret void

define void @captured2() nounwind uwtable sanitize_thread {
entry:
  %ptr = alloca i32, align 4
  %tmp = alloca i32*, align 8
  ; transitive escape
  store i32* %ptr, i32** %tmp, align 8
  %0 = load i32*, i32** %tmp, align 8
  store i32* %0, i32** @sink, align 8
  store i32 42, i32* %ptr, align 4
  ret void
}
; CHECK-LABEL: define void @captured2
; CHECK: __tsan_write
; CHECK: __tsan_write
; CHECK: ret void

define void @notcaptured0() nounwind uwtable sanitize_thread {
entry:
  %ptr = alloca i32, align 4
  store i32 42, i32* %ptr, align 4
  ; escapes due to call
  call void @escape(i32* %ptr)
  ret void
}
; CHECK-LABEL: define void @notcaptured0
; CHECK: __tsan_write
; CHECK: ret void

define void @notcaptured1() nounwind uwtable sanitize_thread {
entry:
  %ptr = alloca i32, align 4
  store i32 42, i32* %ptr, align 4
  ; escapes due to store into global
  store i32* %ptr, i32** @sink, align 8
  ret void
}
; CHECK-LABEL: define void @notcaptured1
; CHECK: __tsan_write
; CHECK: __tsan_write
; CHECK: ret void

define void @notcaptured2() nounwind uwtable sanitize_thread {
entry:
  %ptr = alloca i32, align 4
  %tmp = alloca i32*, align 8
  store i32 42, i32* %ptr, align 4
  ; transitive escape
  store i32* %ptr, i32** %tmp, align 8
  %0 = load i32*, i32** %tmp, align 8
  store i32* %0, i32** @sink, align 8
  ret void
}
; CHECK-LABEL: define void @notcaptured2
; CHECK: __tsan_write
; CHECK: __tsan_write
; CHECK: ret void


