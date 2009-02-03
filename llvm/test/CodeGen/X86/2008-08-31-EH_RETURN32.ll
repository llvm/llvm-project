; Check that eh_return & unwind_init were properly lowered
; RUN: llvm-as < %s | llc | grep %ebp | count 9
; RUN: llvm-as < %s | llc | grep %ecx | count 5

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i386-pc-linux"

define i8* @test(i32 %a, i8* %b)  {
entry:
  call void @llvm.eh.unwind.init()
  %foo   = alloca i32
  call void @llvm.eh.return.i32(i32 %a, i8* %b)
  unreachable
}

declare void @llvm.eh.return.i32(i32, i8*)
declare void @llvm.eh.unwind.init()
