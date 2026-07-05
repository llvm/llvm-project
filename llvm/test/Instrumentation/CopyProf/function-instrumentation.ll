; Tests basic CopyProf instrumentation of special member functions.
;
; RUN: opt < %s -passes='function(copyprof),module(copyprof-module)' -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

;; Verifies that the module constructor and global ctors are set up.
; CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }]
; CHECK-SAME: i32 0, ptr @copyprof.module_ctor

;; Tests constructor instrumentation (1-arg: this).
define void @ctor(ptr %this) "copyprof-ctor"="24" {
entry:
  %field = getelementptr i8, ptr %this, i64 8
  store i32 0, ptr %field
  ret void
}
; CHECK-LABEL: define void @ctor(ptr %this)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @__copyprof_ctor_enter_callback(ptr %this, i64 24)
; CHECK-NOT:     call void @__copyprof_ctor_enter_callback
; CHECK:         call void @__copyprof_ctor_exit_callback(ptr %this, i64 24)
; CHECK-NEXT:    ret void

;; Tests copy constructor instrumentation (2-arg: this, other).
define void @copy_ctor(ptr %this, ptr %other) "copyprof-copy-ctor"="16" {
entry:
  %val = load i32, ptr %other
  store i32 %val, ptr %this
  ret void
}
; CHECK-LABEL: define void @copy_ctor(ptr %this, ptr %other)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @__copyprof_copy_ctor_enter_callback(ptr %this, ptr %other, i64 16)
; CHECK-NOT:     call void @__copyprof_copy_ctor_enter_callback
; CHECK:         call void @__copyprof_copy_ctor_exit_callback(ptr %this, ptr %other, i64 16)
; CHECK-NEXT:    ret void

;; Tests copy assignment operator instrumentation (2-arg: this, other).
define void @copy_assign(ptr %this, ptr %other) "copyprof-copy-assign-op"="32" {
entry:
  %val = load i64, ptr %other
  store i64 %val, ptr %this
  ret void
}
; CHECK-LABEL: define void @copy_assign(ptr %this, ptr %other)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @__copyprof_copy_assign_op_enter_callback(ptr %this, ptr %other, i64 32)
; CHECK-NOT:     call void @__copyprof_copy_assign_op_enter_callback
; CHECK:         call void @__copyprof_copy_assign_op_exit_callback(ptr %this, ptr %other, i64 32)
; CHECK-NEXT:    ret void

;; Tests destructor instrumentation (1-arg: this).
define void @dtor(ptr %this) "copyprof-dtor"="24" {
entry:
  ret void
}
; CHECK-LABEL: define void @dtor(ptr %this)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @__copyprof_dtor_enter_callback(ptr %this, i64 24)
; CHECK-NOT:     call void @__copyprof_dtor_enter_callback
; CHECK:         call void @__copyprof_dtor_exit_callback(ptr %this, i64 24)
; CHECK-NEXT:    ret void

;; Tests that exit callbacks are inserted before multiple return instructions.
define void @ctor_multi_ret(ptr %this, i1 %cond) "copyprof-ctor"="8" {
entry:
  br i1 %cond, label %then, label %else
then:
  ret void
else:
  ret void
}
; CHECK-LABEL: define void @ctor_multi_ret(ptr %this, i1 %cond)
; CHECK:         call void @__copyprof_ctor_enter_callback(ptr %this, i64 8)
; CHECK-NOT:     call void @__copyprof_ctor_enter_callback
; CHECK:       then:
; CHECK:         call void @__copyprof_ctor_exit_callback(ptr %this, i64 8)
; CHECK-NEXT:    ret void
; CHECK:       else:
; CHECK:         call void @__copyprof_ctor_exit_callback(ptr %this, i64 8)
; CHECK-NEXT:    ret void

;; Tests that exit callbacks are inserted before resume instructions
;; (exception unwinding).
define void @ctor_with_resume(ptr %this) "copyprof-ctor"="8" personality ptr @__gxx_personality_v0 {
entry:
  invoke void @may_throw() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %lp = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %lp
}

declare void @may_throw()
declare i32 @__gxx_personality_v0(...)

; CHECK-LABEL: define void @ctor_with_resume(ptr %this)
; CHECK:         call void @__copyprof_ctor_enter_callback(ptr %this, i64 8)
; CHECK:       cont:
; CHECK:         call void @__copyprof_ctor_exit_callback(ptr %this, i64 8)
; CHECK-NEXT:    ret void
; CHECK:       lpad:
; CHECK:         call void @__copyprof_ctor_exit_callback(ptr %this, i64 8)
; CHECK-NEXT:    resume

;; Tests that a function containing a musttail call is skipped entirely.
declare void @tail_callee(ptr)

define void @ctor_with_musttail(ptr %this) "copyprof-ctor"="8" {
entry:
  musttail call void @tail_callee(ptr %this)
  ret void
}
; CHECK-LABEL: define void @ctor_with_musttail(ptr %this)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    musttail call void @tail_callee(ptr %this)
; CHECK-NEXT:    ret void
; CHECK-NOT:     call void @__copyprof_

;; Tests that entry callbacks are inserted after alloca instructions.
define void @ctor_with_alloca(ptr %this) "copyprof-ctor"="8" {
entry:
  %tmp = alloca i32
  store i32 0, ptr %tmp
  ret void
}
; CHECK-LABEL: define void @ctor_with_alloca(ptr %this)
; CHECK:         %tmp = alloca i32
; CHECK-NEXT:    call void @__copyprof_ctor_enter_callback(ptr %this, i64 8)
; CHECK:         call void @__copyprof_ctor_exit_callback(ptr %this, i64 8)
; CHECK-NEXT:    ret void

;; Tests that no exit callback is inserted before unreachable terminators.
define void @ctor_with_unreachable(ptr %this) "copyprof-ctor"="8" {
entry:
  call void @may_throw()
  unreachable
}
; CHECK-LABEL: define void @ctor_with_unreachable(ptr %this)
; CHECK:         call void @__copyprof_ctor_enter_callback(ptr %this, i64 8)
; CHECK-NOT:     call void @__copyprof_ctor_exit_callback
; CHECK:         unreachable

;; Verifies that the module constructor calls the init function and is marked so
;; that it's never instrumented itself.
; CHECK: define internal void @copyprof.module_ctor()
; CHECK:   call void @__copyprof_init()
; CHECK: disable_sanitizer_instrumentation
