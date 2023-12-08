; RUN: opt < %s -passes=instcombine -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare i32 @__gxx_personality_v0(...)
declare void @__cxa_call_unexpected(ptr)
declare i64 @llvm.objectsize.i64(ptr, i1) nounwind readonly
declare ptr @_Znwm(i64)


; CHECK-LABEL: @f1(
define i64 @f1() nounwind uwtable ssp personality ptr @__gxx_personality_v0 {
entry:
; CHECK: nvoke noalias ptr undef()
  %call = invoke noalias ptr undef()
          to label %invoke.cont unwind label %lpad

invoke.cont:
; CHECK: ret i64 0
  %0 = tail call i64 @llvm.objectsize.i64(ptr %call, i1 false)
  ret i64 %0

lpad:
  %1 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  %2 = extractvalue { ptr, i32 } %1, 0
  tail call void @__cxa_call_unexpected(ptr %2) noreturn nounwind
  unreachable
}

; CHECK-LABEL: @f2(
define i64 @f2() nounwind uwtable ssp personality ptr @__gxx_personality_v0 {
entry:
; CHECK: nvoke noalias ptr null()
  %call = invoke noalias ptr null()
          to label %invoke.cont unwind label %lpad

invoke.cont:
; CHECK: ret i64 0
  %0 = tail call i64 @llvm.objectsize.i64(ptr %call, i1 false)
  ret i64 %0

lpad:
  %1 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  %2 = extractvalue { ptr, i32 } %1, 0
  tail call void @__cxa_call_unexpected(ptr %2) noreturn nounwind
  unreachable
}

; CHECK-LABEL: @f2_no_null_opt(
define i64 @f2_no_null_opt() nounwind uwtable ssp #0 personality ptr @__gxx_personality_v0 {
entry:
; CHECK: invoke noalias ptr null()
  %call = invoke noalias ptr null()
          to label %invoke.cont unwind label %lpad

invoke.cont:
; CHECK: call i64 @llvm.objectsize.i64.p0(ptr %call, i1 false, i1 false, i1 false)
  %0 = tail call i64 @llvm.objectsize.i64(ptr %call, i1 false)
  ret i64 %0

lpad:
  %1 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  %2 = extractvalue { ptr, i32 } %1, 0
  tail call void @__cxa_call_unexpected(ptr %2) noreturn nounwind
  unreachable
}
attributes #0 = { null_pointer_is_valid }

; CHECK-LABEL: @f3(
define void @f3() nounwind uwtable ssp personality ptr @__gxx_personality_v0 {
; CHECK: invoke void @llvm.donothing()
  %call = invoke noalias ptr @_Znwm(i64 13)
          to label %invoke.cont unwind label %lpad

invoke.cont:
  ret void

lpad:
  %1 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  %2 = extractvalue { ptr, i32 } %1, 0
  tail call void @__cxa_call_unexpected(ptr %2) noreturn nounwind
  unreachable
}
