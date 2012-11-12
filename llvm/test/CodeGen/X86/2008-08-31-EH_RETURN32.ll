; Check that eh_return & unwind_init were properly lowered
; RUN: llc < %s -verify-machineinstrs | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i386-pc-linux"

; CHECK: test1
; CHECK: pushl %ebp
define i8* @test1(i32 %a, i8* %b)  {
entry:
  call void @llvm.eh.unwind.init()
  %foo   = alloca i32
  call void @llvm.eh.return.i32(i32 %a, i8* %b)
; CHECK: movl 12(%ebp), %[[ECX:e..]]
; CHECK: movl 8(%ebp), %[[EAX:e..]]
; CHECK: movl %[[ECX]], 4(%ebp,%[[EAX]])
; CHECK: leal 4(%ebp,%[[EAX]]), %[[ECX2:e..]]
; CHECK: movl %[[ECX2]], %esp
; CHECK: ret
  unreachable
}

; CHECK: test2
; CHECK: pushl %ebp
define i8* @test2(i32 %a, i8* %b)  {
entry:
  call void @llvm.eh.return.i32(i32 %a, i8* %b)
; CHECK: movl 12(%ebp), %[[ECX:e..]]
; CHECK: movl 8(%ebp), %[[EAX:e..]]
; CHECK: movl %[[ECX]], 4(%ebp,%[[EAX]])
; CHECK: leal 4(%ebp,%[[EAX]]), %[[ECX2:e..]]
; CHECK: movl %[[ECX2]], %esp
; CHECK: ret
  unreachable
}

declare void @llvm.eh.return.i32(i32, i8*)
declare void @llvm.eh.unwind.init()
