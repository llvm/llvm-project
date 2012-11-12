; Check that eh_return & unwind_init were properly lowered
; RUN: llc < %s | grep %rbp | count 7
; RUN: llc < %s | grep %rcx | count 3

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define i8* @test(i64 %a, i8* %b)  {
entry:
  call void @llvm.eh.unwind.init()
  %foo   = alloca i32
  call void @llvm.eh.return.i64(i64 %a, i8* %b)
  unreachable
}

declare void @llvm.eh.return.i64(i64, i8*)
declare void @llvm.eh.unwind.init()
