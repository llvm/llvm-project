; RUN: opt -S -globaldce -sjljehprepare < %s | FileCheck %s
target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7--ios5.0.0"

define void @g() personality i32 (...)* @__gxx_personality_sj0 {
entry:
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  invoke void @f()
          to label %try.cont unwind label %lpad

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          cleanup
  br label %try.cont

try.cont:                                         ; preds = %catch, %invoke.cont
  ret void
}

declare void @f()

declare i32 @__gxx_personality_sj0(...)

; CHECK-LABEL: define void @g(
; CHECK: call void @llvm.eh.sjlj.callsite(
; CHECK: call void @_Unwind_SjLj_Register(
; CHECK: invoke void @f(
; CHECK: landingpad
; CHECK-NEXT: cleanup
; CHECK: call void @_Unwind_SjLj_Unregister(
