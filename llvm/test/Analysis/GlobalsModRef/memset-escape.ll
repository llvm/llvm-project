; RUN: opt < %s -O1 -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

@a = internal global [3 x i32] zeroinitializer, align 4
@b = common global i32 0, align 4

; The important thing we're checking for here is the reload of (some element of)
; @a after the memset.

; CHECK-LABEL: @main
; CHECK: call void @llvm.memset.p0i8.i64{{.*}} @a
; CHECK: store i32 3
; CHECK: load i32, i32* getelementptr {{.*}} @a
; CHECK: icmp eq i32
; CHECK: br i1

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  %c = alloca [1 x i32], align 4
  store i32 0, i32* %retval, align 4
  %0 = bitcast [1 x i32]* %c to i8*
  call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 4, i32 4, i1 false)
  store i32 1, i32* getelementptr inbounds ([3 x i32], [3 x i32]* @a, i64 0, i64 2), align 4
  store i32 0, i32* @b, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, i32* @b, align 4
  %cmp = icmp slt i32 %1, 3
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i32, i32* @b, align 4
  %idxprom = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds [3 x i32], [3 x i32]* @a, i64 0, i64 %idxprom
  store i32 0, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %3 = load i32, i32* @b, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, i32* @b, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %4 = load i32, i32* getelementptr inbounds ([3 x i32], [3 x i32]* @a, i64 0, i64 2), align 4
  %cmp1 = icmp ne i32 %4, 0
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.end
  call void @abort() #3
  unreachable

if.end:                                           ; preds = %for.end
  ret i32 0
}

; Function Attrs: nounwind argmemonly
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind argmemonly

; Function Attrs: noreturn nounwind
declare void @abort() noreturn nounwind
