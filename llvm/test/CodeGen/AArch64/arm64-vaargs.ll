; RUN: llc < %s -mtriple=arm64-apple-darwin11.0.0 | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32:64"

define float @t1(i8* nocapture %fmt, ...) nounwind ssp {
entry:
; CHECK: t1
; CHECK: fcvt
  %argp = alloca i8*, align 8
  %argp1 = bitcast i8** %argp to i8*
  call void @llvm.va_start(i8* %argp1)
  %0 = va_arg i8** %argp, i32
  %1 = va_arg i8** %argp, float
  call void @llvm.va_end(i8* %argp1)
  ret float %1
}

declare void @llvm.va_start(i8*) nounwind

declare void @llvm.va_end(i8*) nounwind
