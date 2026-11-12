; RUN: opt < %s -passes=dfsan -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @__dfsan_arg_tls
; CHECK: = external thread_local(initialexec) global [100 x i64]

; CHECK-LABEL: @__dfsan_retval_tls
; CHECK: = external thread_local(initialexec) global [100 x i64]

declare i32 @f(i32)
declare float @llvm.sqrt.f32(float)

; CHECK-LABEL: @call.dfsan
define i32 @call() {
  ; CHECK: store{{.*}}__dfsan_arg_tls
  ; CHECK: call{{.*}}@f.dfsan
  ; CHECK: load{{.*}}__dfsan_retval_tls
  %r = call i32 @f(i32 0)

  ; CHECK-NOT: store{{.*}}__dfsan_arg_tls
  %i = call float @llvm.sqrt.f32(float -1.0)

  ; CHECK: store{{.*}}__dfsan_retval_tls
  ; CHECK: ret i32
  ret i32 %r
}

declare i32 @__gxx_personality_v0(...)

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()

declare void @g(...)

; CHECK-LABEL: @h.dfsan
; CHECK: personality {{.*}} @__gxx_personality_v0.dfsan {
define i32 @h() personality ptr @__gxx_personality_v0 {
entry:
; CHECK: invoke void (...) @g.dfsan(i32 42)
  invoke void (...) @g(i32 42)
          to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0

  ; CHECK: store {{.*}} @__dfsan_arg_tls
  ; CHECK: call {{.*}} @__cxa_begin_catch.dfsan
  ; CHECK: load {{.*}} @__dfsan_retval_tls
  %2 = tail call ptr @__cxa_begin_catch(ptr %1)

  ; CHECK: call {{.*}} @__cxa_end_catch.dfsan
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  ret i32 0
}
