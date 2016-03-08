; RUN: opt < %s -dfsan -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @__dfsan_arg_tls
; CHECK: = external thread_local(initialexec) global [64 x i16]

; CHECK-LABEL: @__dfsan_retval_tls
; CHECK: = external thread_local(initialexec) global i16

declare i32 @f(i32)
declare float @llvm.sqrt.f32(float)

; CHECK-LABEL: @"dfs$call"
define i32 @call() {
  ; CHECK: store{{.*}}__dfsan_arg_tls
  ; CHECK: call{{.*}}@"dfs$f"
  ; CHECK: load{{.*}}__dfsan_retval_tls
  %r = call i32 @f(i32 0)

  ; CHECK-NOT: store{{.*}}__dfsan_arg_tls
  %i = call float @llvm.sqrt.f32(float -1.0)

  ; CHECK: store{{.*}}__dfsan_retval_tls
  ; CHECK: ret i32
  ret i32 %r
}

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

declare void @g(...)

; CHECK-LABEL: @"dfs$h"
; CHECK: personality {{.*}} @"dfs$__gxx_personality_v0" {{.*}} {
define i32 @h() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
; CHECK: invoke void (...) @"dfs$g"(i32 42)
  invoke void (...) @g(i32 42)
          to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0

  ; CHECK: store {{.*}} @__dfsan_arg_tls
  ; CHECK: call {{.*}} @"dfs$__cxa_begin_catch"
  ; CHECK: load {{.*}} @__dfsan_retval_tls
  %2 = tail call i8* @__cxa_begin_catch(i8* %1)

  ; CHECK: call {{.*}} @"dfs$__cxa_end_catch"
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  ret i32 0
}
