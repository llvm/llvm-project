; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

define void @func(i8* %i) nounwind ssp {
; CHECK: @func
; CHECK: @__strcpy_chk(i8* %arraydecay, i8* %i, i64 32)
entry:
  %s = alloca [32 x i8], align 16
  %arraydecay = getelementptr inbounds [32 x i8]* %s, i32 0, i32 0
  %call = call i8* @__strcpy_chk(i8* %arraydecay, i8* %i, i64 32)
  call void @func2(i8* %arraydecay)
  ret void
}

declare i8* @__strcpy_chk(i8*, i8*, i64) nounwind

declare void @func2(i8*)
