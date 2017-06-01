; RUN: opt -instcombine -S < %s | FileCheck %s

; #include <stdlib.h>
; #include <stdio.h>
;
; int foo1(int N) {
;   char Big[20];
;   char Small[10];
;   char *Ptr = N ? Big + 10 : Small;
;   return __builtin_object_size(Ptr, 0);
; }
;
; void foo() {
;   size_t ret;
;   ret = foo1(0);
;   printf("\n %d", ret);
; }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [5 x i8] c"\0A %d\00", align 1

define i32 @foo1(i32 %N) {
entry:
  %Big = alloca [20 x i8], align 16
  %Small = alloca [10 x i8], align 1
  %0 = getelementptr inbounds [20 x i8], [20 x i8]* %Big, i64 0, i64 0
  call void @llvm.lifetime.start(i64 20, i8* %0)
  %1 = getelementptr inbounds [10 x i8], [10 x i8]* %Small, i64 0, i64 0
  call void @llvm.lifetime.start(i64 10, i8* %1)
  %tobool = icmp ne i32 %N, 0
  %add.ptr = getelementptr inbounds [20 x i8], [20 x i8]* %Big, i64 0, i64 10
  %cond = select i1 %tobool, i8* %add.ptr, i8* %1
  %2 = call i64 @llvm.objectsize.i64.p0i8(i8* %cond, i1 false)
  %conv = trunc i64 %2 to i32
  call void @llvm.lifetime.end(i64 10, i8* %1)
  call void @llvm.lifetime.end(i64 20, i8* %0)
  ret i32 %conv
; CHECK: ret i32 10 
}

declare void @llvm.lifetime.start(i64, i8* nocapture)

declare i64 @llvm.objectsize.i64.p0i8(i8*, i1)

declare void @llvm.lifetime.end(i64, i8* nocapture)

define void @foo() {
entry:
  %call = tail call i32 @foo1(i32 0)
  %conv = sext i32 %call to i64
  %call1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0), i64 %conv)
  ret void
}

declare i32 @printf(i8* nocapture readonly, ...)

