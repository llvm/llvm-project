; RUN: opt < %s -passes=asan -asan-stack-dynamic-alloca -asan-use-stack-safety=0 -asan-use-after-return=runtime -S | FileCheck %s --check-prefixes=CHECK,CHECK-RUNTIME
; RUN: opt < %s -passes=asan -asan-stack-dynamic-alloca -asan-mapping-scale=5 -asan-use-stack-safety=0 -asan-use-after-return=runtime -S | FileCheck %s --check-prefixes=CHECK,CHECK-RUNTIME
; RUN: opt < %s -passes=asan  -asan-stack-dynamic-alloca -asan-use-stack-safety=0 -asan-use-after-return=always -S | FileCheck %s --check-prefixes=CHECK,CHECK-ALWAYS --implicit-check-not=__asan_option_detect_stack_use_after_return
; RUN: opt < %s -passes=asan -asan-stack-dynamic-alloca -asan-use-stack-safety=0 -asan-use-after-return=always -S | FileCheck %s --check-prefixes=CHECK,CHECK-ALWAYS --implicit-check-not=__asan_option_detect_stack_use_after_return
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @Func1() sanitize_address {
entry:
; CHECK-LABEL: Func1

; CHECK: entry:
; CHECK-RUNTIME: load i32, ptr @__asan_option_detect_stack_use_after_return
; COM: CHECK-NORUNTIME-NOT: load i32, ptr @__asan_option_detect_stack_use_after_return

; CHECK-RUNTIME: [[UAR_ENABLED_BB:^[0-9]+]]:
; CHECK-RUNTIME: [[FAKE_STACK_RT:%[0-9]+]] = call i64 @__asan_stack_malloc_
; CHECK-ALWAYS: [[FAKE_STACK_RT:%[0-9]+]] = call i64 @__asan_stack_malloc_always_

; CHECK-RUNTIME: [[FAKE_STACK_BB:^[0-9]+]]:
; CHECK-RUNTIME: [[FAKE_STACK:%[0-9]+]] = phi i64 [ 0, %entry ], [ [[FAKE_STACK_RT]], %[[UAR_ENABLED_BB]] ]
; CHECK-RUNTIME: [[FAKE_STACK_PTR:%[0-9]+]] = inttoptr i64 [[FAKE_STACK]] to ptr
; CHECK-ALWAYS: [[FAKE_STACK_PTR:%[0-9]+]] = inttoptr i64 [[FAKE_STACK_RT]] to ptr
; CHECK-RUNTIME: icmp eq i64 [[FAKE_STACK]], 0
; CHECK-ALWAYS: icmp eq i64 [[FAKE_STACK_RT]], 0

; CHECK: [[NO_FAKE_STACK_BB:^[0-9]+]]:
; CHECK: %MyAlloca = alloca i8, i64

; CHECK-RUNTIME: phi ptr [ [[FAKE_STACK_PTR]], %[[FAKE_STACK_BB]] ], [ %MyAlloca, %[[NO_FAKE_STACK_BB]] ]
; CHECK-ALWAYS: phi ptr [ [[FAKE_STACK_PTR]], %entry ], [ %MyAlloca, %[[NO_FAKE_STACK_BB]] ]

; CHECK: ret void

  %XXX = alloca [20 x i8], align 1
  store volatile i8 0, ptr %XXX
  ret void
}

; Test that dynamic alloca is not used for functions with inline assembly.
define void @Func2() sanitize_address {
entry:
; CHECK-LABEL: Func2
; CHECK: alloca [96 x i8]
; CHECK: ret void

  %XXX = alloca [20 x i8], align 1
  store volatile i8 0, ptr %XXX
  call void asm sideeffect "mov %%rbx, %%rcx", "~{dirflag},~{fpsr},~{flags}"() nounwind
  ret void
}

; Test that dynamic alloca is not used when setjmp is present.
%struct.__jmp_buf_tag = type { [8 x i64], i32, %struct.__sigset_t }
%struct.__sigset_t = type { [16 x i64] }
@_ZL3buf = internal global [1 x %struct.__jmp_buf_tag] zeroinitializer, align 16

define void @Func3() uwtable sanitize_address {
; CHECK-LABEL: define void @Func3
; CHECK-NOT: __asan_option_detect_stack_use_after_return
; CHECK-NOT: __asan_stack_malloc
; CHECK: call void @__asan_handle_no_return
; CHECK: call void @longjmp
; CHECK: ret void
entry:
  %a = alloca i32, align 4
  %call = call i32 @_setjmp(ptr @_ZL3buf) nounwind returns_twice
  %cmp = icmp eq i32 0, %call
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @longjmp(ptr @_ZL3buf, i32 1) noreturn nounwind
  unreachable

if.end:                                           ; preds = %entry
  call void @_Z10escape_ptrPi(ptr %a)
  ret void
}

declare i32 @_setjmp(ptr) nounwind returns_twice
declare void @longjmp(ptr, i32) noreturn nounwind
declare void @_Z10escape_ptrPi(ptr)
