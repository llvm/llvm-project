; RUN: opt < %s -instcombine -S | not grep call
; RUN: opt < %s -O3 -S | not grep xyz
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

@.str = internal constant [4 x i8] c"xyz\00"		; <[4 x i8]*> [#uses=1]

define void @foo(i8* %P) {
entry:
  %P_addr = alloca i8*
  store i8* %P, i8** %P_addr
  %tmp = load i8*, i8** %P_addr, align 4
  %tmp1 = getelementptr [4 x i8], [4 x i8]* @.str, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %tmp, i8* %tmp1, i32 4, i1 false)
  br label %return

return:                                           ; preds = %entry
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind
