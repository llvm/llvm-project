; PR12202
; RUN: opt < %s -scalarrepl -S
; Ensure that we do not hang or crash when feeding a negative value to memset

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f80:128:128-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S32"
target triple = "i686-pc-win32"

define i32 @test() nounwind {
entry:
  %retval = alloca i32, align 4
  %buff = alloca [1 x i8], align 1
  store i32 0, i32* %retval
  %0 = bitcast [1 x i8]* %buff to i8*
  call void @llvm.memset.p0i8.i32(i8* %0, i8 0, i32 1, i32 1, i1 false)
  %arraydecay = getelementptr inbounds [1 x i8]* %buff, i32 0, i32 0
  call void @llvm.memset.p0i8.i32(i8* %arraydecay, i8 -1, i32 -8, i32 1, i1 false)	; Negative 8!
  ret i32 0
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind
