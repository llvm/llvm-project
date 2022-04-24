; RUN: opt < %s -passes=tsan -S | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-EXC
; RUN: opt < %s -passes=tsan -S -tsan-handle-cxx-exceptions=0 | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NOEXC

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare void @can_throw()
declare void @cannot_throw() nounwind

define i32 @func1() sanitize_thread {
  call void @can_throw()
  ret i32 0
  ; CHECK-EXC: define i32 @func1()
  ; CHECK-EXC:   call void @__tsan_func_entry
  ; CHECK-EXC:   invoke void @can_throw()
  ; CHECK-EXC: .noexc:
  ; CHECK-EXC:   call void @__tsan_func_exit()
  ; CHECK-EXC:   ret i32 0
  ; CHECK-EXC: tsan_cleanup:
  ; CHECK-EXC:   call void @__tsan_func_exit()
  ; CHECK-EXC:   resume
  ; CHECK-NOEXC: define i32 @func1()
  ; CHECK-NOEXC: call void @__tsan_func_entry
  ; CHECK-NOEXC: call void @can_throw()
  ; CHECK-NOEXC: call void @__tsan_func_exit()
  ; CHECK-NOEXC: ret i32 0
}

define i32 @func2() sanitize_thread {
  call void @cannot_throw()
  ret i32 0
  ; CHECK: define i32 @func2()
  ; CHECK: call void @__tsan_func_entry
  ; CHECK: call void @cannot_throw()
  ; CHECK: call void @__tsan_func_exit()
  ; CHECK: ret i32 0
}

define i32 @func3(i32* %p) sanitize_thread {
  %a = load i32, i32* %p
  ret i32 %a
  ; CHECK: define i32 @func3(i32* %p)
  ; CHECK: call void @__tsan_func_entry
  ; CHECK: call void @__tsan_read4
  ; CHECK: %a = load i32, i32* %p
  ; CHECK: call void @__tsan_func_exit()
  ; CHECK: ret i32 %a
}

define i32 @func4() sanitize_thread nounwind {
  call void @can_throw()
  ret i32 0
  ; CHECK: define i32 @func4()
  ; CHECK: call void @__tsan_func_entry
  ; CHECK: call void @can_throw()
  ; CHECK: call void @__tsan_func_exit()
  ; CHECK: ret i32 0
}
