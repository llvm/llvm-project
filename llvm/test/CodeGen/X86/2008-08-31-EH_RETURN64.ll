; Check that eh_return & unwind_init were properly lowered
; RUN: llc < %s -verify-machineinstrs | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: test
; CHECK: pushq %rbp
; CHECK: movq %rsp, %rbp
; CHECK: popq %rbp
; CHECK: movq %rcx, %rsp
; CHECK: retq # eh_return, addr: %rcx
define ptr @test(i64 %a, ptr %b)  {
entry:
  call void @llvm.eh.unwind.init()
  %foo   = alloca i32
  call void @llvm.eh.return.i64(i64 %a, ptr %b)
  unreachable
}

declare void @llvm.eh.return.i64(i64, ptr)
declare void @llvm.eh.unwind.init()

@b = common global i32 0, align 4
@a = common global i32 0, align 4

; PR14750
; This function contains a normal return as well as eh_return.
; CHECK: _Unwind_Resume_or_Rethrow
define i32 @_Unwind_Resume_or_Rethrow() nounwind uwtable ssp {
entry:
  %0 = load i32, ptr @b, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  ret i32 0

if.end:                                           ; preds = %entry
  %call = tail call i32 (...) @_Unwind_ForcedUnwind_Phase2() nounwind
  store i32 %call, ptr @a, align 4
  %tobool1 = icmp eq i32 %call, 0
  br i1 %tobool1, label %cond.end, label %cond.true

cond.true:                                        ; preds = %if.end
  tail call void @abort() noreturn nounwind
  unreachable

cond.end:                                         ; preds = %if.end
  tail call void @llvm.eh.return.i64(i64 0, ptr null)
  unreachable
}

declare i32 @_Unwind_ForcedUnwind_Phase2(...)
declare void @abort() noreturn
