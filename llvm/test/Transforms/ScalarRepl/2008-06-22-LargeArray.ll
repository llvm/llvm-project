; RUN: opt < %s -scalarrepl -S | grep "call.*mem" 
; PR2369

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

define void @memtest1(i8* %dst, i8* %src) nounwind {
entry:
  %temp = alloca [200 x i8]
  %temp1 = bitcast [200 x i8]* %temp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %temp1, i8* %src, i32 200, i32 1, i1 false)
  %temp3 = bitcast [200 x i8]* %temp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dst, i8* %temp3, i32 200, i32 1, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
