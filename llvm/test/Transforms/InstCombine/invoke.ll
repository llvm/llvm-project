; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare i32 @__gxx_personality_v0(...)
declare void @__cxa_call_unexpected(i8*)
declare i64 @llvm.objectsize.i64(i8*, i1) nounwind readonly
declare i8* @_Znwm(i64)


; CHECK-LABEL: @f1(
define i64 @f1() nounwind uwtable ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
; CHECK: nvoke noalias i8* undef()
  %call = invoke noalias i8* undef()
          to label %invoke.cont unwind label %lpad

invoke.cont:
; CHECK: ret i64 0
  %0 = tail call i64 @llvm.objectsize.i64(i8* %call, i1 false)
  ret i64 %0

lpad:
  %1 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  %2 = extractvalue { i8*, i32 } %1, 0
  tail call void @__cxa_call_unexpected(i8* %2) noreturn nounwind
  unreachable
}

; CHECK-LABEL: @f2(
define i64 @f2() nounwind uwtable ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
; CHECK: nvoke noalias i8* null()
  %call = invoke noalias i8* null()
          to label %invoke.cont unwind label %lpad

invoke.cont:
; CHECK: ret i64 0
  %0 = tail call i64 @llvm.objectsize.i64(i8* %call, i1 false)
  ret i64 %0

lpad:
  %1 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  %2 = extractvalue { i8*, i32 } %1, 0
  tail call void @__cxa_call_unexpected(i8* %2) noreturn nounwind
  unreachable
}

; CHECK-LABEL: @f2_no_null_opt(
define i64 @f2_no_null_opt() nounwind uwtable ssp #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
; CHECK: invoke noalias i8* null()
  %call = invoke noalias i8* null()
          to label %invoke.cont unwind label %lpad

invoke.cont:
; CHECK: call i64 @llvm.objectsize.i64.p0i8(i8* %call, i1 false, i1 false, i1 false)
  %0 = tail call i64 @llvm.objectsize.i64(i8* %call, i1 false)
  ret i64 %0

lpad:
  %1 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  %2 = extractvalue { i8*, i32 } %1, 0
  tail call void @__cxa_call_unexpected(i8* %2) noreturn nounwind
  unreachable
}
attributes #0 = { "null-pointer-is-valid"="true" }

; CHECK-LABEL: @f3(
define void @f3() nounwind uwtable ssp personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK: invoke void @llvm.donothing()
  %call = invoke noalias i8* @_Znwm(i64 13)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret void

lpad:
  %1 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  %2 = extractvalue { i8*, i32 } %1, 0
  tail call void @__cxa_call_unexpected(i8* %2) noreturn nounwind
  unreachable
}
