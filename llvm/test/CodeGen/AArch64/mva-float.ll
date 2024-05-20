; RUN: llc < %s -march=aarch64 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "arm64"

%struct.__va_list = type { i8*, i8*, i8*, i32, i32 }

declare i32 @vfunc(i8*, i8*)
declare void @llvm.va_start(i8*)
declare void @llvm.va_end(i8*)

define i32 @func(i8*, double, ...) {
entry:
  %argp = alloca %struct.__va_list, align 8
  %argp1 = bitcast %struct.__va_list* %argp to i8*
  call void @llvm.va_start(i8* %argp1)
; CHECK: {{stp.*q[0-9]+}}
  %ret = call i32 @vfunc(i8* %0, i8* %argp1)
  call void @llvm.va_end(i8* %argp1)
  ret i32 %ret
}
