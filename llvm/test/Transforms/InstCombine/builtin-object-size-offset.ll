; RUN: opt -passes=instcombine -S < %s | FileCheck %s

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
  call void @llvm.lifetime.start.p0(ptr %Big)
  call void @llvm.lifetime.start.p0(ptr %Small)
  %tobool = icmp ne i32 %N, 0
  %add.ptr = getelementptr inbounds [20 x i8], ptr %Big, i64 0, i64 10
  %cond = select i1 %tobool, ptr %add.ptr, ptr %Small
  %0 = call i64 @llvm.objectsize.i64.p0(ptr %cond, i1 false)
  %conv = trunc i64 %0 to i32
  call void @llvm.lifetime.end.p0(ptr %Small)
  call void @llvm.lifetime.end.p0(ptr %Big)
  ret i32 %conv
; CHECK: ret i32 10 
}

declare void @llvm.lifetime.start.p0(ptr nocapture)

declare i64 @llvm.objectsize.i64.p0(ptr, i1)

declare void @llvm.lifetime.end.p0(ptr nocapture)

define void @foo() {
entry:
  %call = tail call i32 @foo1(i32 0)
  %conv = sext i32 %call to i64
  %call1 = tail call i32 (ptr, ...) @printf(ptr @.str, i64 %conv)
  ret void
}

declare i32 @printf(ptr nocapture readonly, ...)

