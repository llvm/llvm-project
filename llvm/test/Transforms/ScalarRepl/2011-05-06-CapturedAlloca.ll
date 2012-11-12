; RUN: opt < %s -instcombine -S | FileCheck %s
; PR9820

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

@func_1.l_10 = internal unnamed_addr constant [4 x i32] [i32 1, i32 0, i32 0, i32 0], align 16

define i32* @noop(i32* %p_29) nounwind readnone {
entry:
  ret i32* %p_29
}

define i32 @main() nounwind {
entry:
  %l_10 = alloca [4 x i32], align 16
  %tmp = bitcast [4 x i32]* %l_10 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp, i8* bitcast ([4 x i32]* @func_1.l_10 to i8*), i64 16, i32 16, i1 false)
; CHECK: call void @llvm.memcpy
  %arrayidx = getelementptr inbounds [4 x i32]* %l_10, i64 0, i64 0
  %call = call i32* @noop(i32* %arrayidx)
  store i32 0, i32* %call
  ret i32 0
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind
