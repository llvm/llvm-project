; RUN: opt < %s -passes=tsan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define i32 @"\01-[NoCalls dealloc]"(i32* %a) "sanitize_thread_no_checking_at_run_time" {
entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}

; CHECK: define i32 @"\01-[NoCalls dealloc]"(i32* %a)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %tmp1 = load i32, i32* %a, align 4
; CHECK-NEXT:   ret i32 %tmp1

declare void @"foo"() nounwind

define i32 @"\01-[WithCalls dealloc]"(i32* %a) "sanitize_thread_no_checking_at_run_time" {
entry:
  %tmp1 = load i32, i32* %a, align 4
  call void @foo()
  ret i32 %tmp1
}

; CHECK: define i32 @"\01-[WithCalls dealloc]"(i32* %a)
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = call i8* @llvm.returnaddress(i32 0)
; CHECK-NEXT:   call void @__tsan_func_entry(i8* %0)
; CHECK-NEXT:   call void @__tsan_ignore_thread_begin()
; CHECK-NEXT:   %tmp1 = load i32, i32* %a, align 4
; CHECK-NEXT:   call void @foo()
; CHECK-NEXT:   call void @__tsan_ignore_thread_end()
; CHECK-NEXT:   call void @__tsan_func_exit()
; CHECK-NEXT:   ret i32 %tmp1
