; RUN: opt < %s -simplifycfg -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare i32 @__gxx_personality_v0(...)
declare void @__cxa_call_unexpected(i8*)
declare void @purefn() nounwind readnone
declare i32 @read_only() nounwind readonly
declare i32 @nounwind_fn() nounwind
declare i32 @fn()


; CHECK-LABEL: @f1(
define i8* @f1() nounwind uwtable ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
; CHECK: call void @llvm.trap()
; CHECK: unreachable
  %call = invoke noalias i8* undef()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i8* %call

lpad:
  %0 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  %1 = extractvalue { i8*, i32 } %0, 0
  tail call void @__cxa_call_unexpected(i8* %1) noreturn nounwind
  unreachable
}

; CHECK-LABEL: @f2(
define i8* @f2() nounwind uwtable ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
; CHECK: call void @llvm.trap()
; CHECK: unreachable
  %call = invoke noalias i8* null()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i8* %call

lpad:
  %0 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  %1 = extractvalue { i8*, i32 } %0, 0
  tail call void @__cxa_call_unexpected(i8* %1) noreturn nounwind
  unreachable
}

; CHECK-LABEL: @f2_no_null_opt(
define i8* @f2_no_null_opt() nounwind uwtable ssp #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
; CHECK: invoke noalias i8* null()
  %call = invoke noalias i8* null()
          to label %invoke.cont unwind label %lpad

; CHECK: invoke.cont:
; CHECK: ret i8* %call
invoke.cont:
  ret i8* %call

lpad:
  %0 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  %1 = extractvalue { i8*, i32 } %0, 0
  tail call void @__cxa_call_unexpected(i8* %1) noreturn nounwind
; CHECK: unreachable
  unreachable
}

; CHECK-LABEL: @f3(
define i32 @f3() nounwind uwtable ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-NEXT: entry
entry:
; CHECK-NEXT: ret i32 3
  %call = invoke i32 @read_only()
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret i32 3

lpad:
  %0 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  %1 = extractvalue { i8*, i32 } %0, 0
  tail call void @__cxa_call_unexpected(i8* %1) noreturn nounwind
  unreachable
}

; CHECK-LABEL: @f4(
define i32 @f4() nounwind uwtable ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-NEXT: entry
entry:
; CHECK-NEXT: call i32 @read_only()
  %call = invoke i32 @read_only()
          to label %invoke.cont unwind label %lpad

invoke.cont:
; CHECK-NEXT: ret i32 %call
  ret i32 %call

lpad:
  %0 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  %1 = extractvalue { i8*, i32 } %0, 0
  tail call void @__cxa_call_unexpected(i8* %1) noreturn nounwind
  unreachable
}

; CHECK-LABEL: @f5(
define i32 @f5(i1 %cond, i8* %a, i8* %b) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  br i1 %cond, label %x, label %y

x:
; CHECK: invoke i32 @fn()
  %call = invoke i32 @fn()
          to label %cont unwind label %lpad

y:
; CHECK: call i32 @nounwind_fn()
  %call2 = invoke i32 @nounwind_fn()
           to label %cont unwind label %lpad

cont:
; CHECK: phi i32
; CHECK: ret i32 %phi
  %phi = phi i32 [%call, %x], [%call2, %y]
  ret i32 %phi

lpad:
; CHECK-NOT: phi
  %phi2 = phi i8* [%a, %x], [%b, %y]
  %0 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
; CHECK: __cxa_call_unexpected(i8* %a)
  tail call void @__cxa_call_unexpected(i8* %phi2) noreturn nounwind
  unreachable
}

; CHECK-LABEL: @f6(
define void @f6() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @purefn()
          to label %invoke.cont1 unwind label %lpad

invoke.cont1:
  %foo = invoke i32 @fn()
          to label %invoke.cont2 unwind label %lpad

invoke.cont2:
  ret void

lpad:
; CHECK-NOT: phi
  %tmp = phi i8* [ null, %invoke.cont1 ], [ null, %entry ]
  landingpad { i8*, i32 }
          cleanup
  ret void
}

attributes #0 = { null_pointer_is_valid }
