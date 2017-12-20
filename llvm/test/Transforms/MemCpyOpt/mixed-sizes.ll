; RUN: opt < %s -memcpyopt -S | FileCheck %s
; Handle memcpy-memcpy dependencies of differing sizes correctly.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Don't delete the second memcpy, even though there's an earlier
; memcpy with a larger size from the same address.

; CHECK-LABEL: @foo
define i32 @foo(i1 %z) {
entry:
  %a = alloca [10 x i32]
  %s = alloca [10 x i32]
  %0 = bitcast [10 x i32]* %a to i8*
  %1 = bitcast [10 x i32]* %s to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull %1, i8 0, i64 40, i32 16, i1 false)
  %arrayidx = getelementptr inbounds [10 x i32], [10 x i32]* %a, i64 0, i64 0
  store i32 1, i32* %arrayidx
  %scevgep = getelementptr [10 x i32], [10 x i32]* %s, i64 0, i64 1
  %scevgep7 = bitcast i32* %scevgep to i8*
  br i1 %z, label %for.body3.lr.ph, label %for.inc7.1

for.body3.lr.ph:                                  ; preds = %entry
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* %scevgep7, i64 17179869180, i32 4, i1 false)
  br label %for.inc7.1

for.inc7.1:
; CHECK: for.inc7.1:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* %scevgep7, i64 4, i32 4, i1 false)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* %scevgep7, i64 4, i32 4, i1 false)
  %2 = load i32, i32* %arrayidx
  ret i32 %2
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i32, i1)
declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i32, i1)
