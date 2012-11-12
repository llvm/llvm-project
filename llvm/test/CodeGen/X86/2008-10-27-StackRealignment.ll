; Linux doesn't support stack realignment for functions with allocas (PR2888).
; Until it does, we shouldn't use movaps to access the stack.  On targets with
; sufficiently aligned stack (e.g. darwin) we should.
; PR8969 - make 32-bit linux have a 16-byte aligned stack
; RUN: llc < %s -mtriple=i386-pc-linux-gnu -mcpu=yonah | grep movaps | count 2
; RUN: llc < %s -mtriple=i686-apple-darwin9 -mcpu=yonah | grep movaps | count 2


target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
  
define void @foo(i32 %t) nounwind {
  %tmp1210 = alloca i8, i32 32, align 4
  call void @llvm.memset.p0i8.i64(i8* %tmp1210, i8 0, i64 32, i32 4, i1 false)
  %x = alloca i8, i32 %t
  call void @dummy(i8* %x)
  ret void
}

declare void @dummy(i8*)

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind
