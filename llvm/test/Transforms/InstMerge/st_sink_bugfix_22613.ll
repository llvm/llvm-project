; ModuleID = 'bug.c'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; RUN: opt -O2 -S < %s | FileCheck %s

; CHECK-LABEL: main
; CHECK: memset
; CHECK: if.then
; CHECK: store
; CHECK: if.end
; CHECK: store
; CHECK: store

@d = common global i32 0, align 4
@b = common global i32 0, align 4
@f = common global [1 x [3 x i8]] zeroinitializer, align 1
@e = common global i32 0, align 4
@c = common global i32 0, align 4
@a = common global i32 0, align 4

; Function Attrs: nounwind uwtable
define void @fn1() {
entry:
  store i32 0, i32* @d, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc8, %entry
  %0 = load i32, i32* @d, align 4
  %cmp = icmp slt i32 %0, 2
  br i1 %cmp, label %for.body, label %for.end10

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* @d, align 4
  %idxprom = sext i32 %1 to i64
  %2 = load i32, i32* @b, align 4
  %idxprom1 = sext i32 %2 to i64
  %arrayidx = getelementptr inbounds [1 x [3 x i8]], [1 x [3 x i8]]* @f, i32 0, i64 %idxprom1
  %arrayidx2 = getelementptr inbounds [3 x i8], [3 x i8]* %arrayidx, i32 0, i64 %idxprom
  store i8 0, i8* %arrayidx2, align 1
  store i32 0, i32* @e, align 4
  br label %for.cond3

for.cond3:                                        ; preds = %for.inc, %for.body
  %3 = load i32, i32* @e, align 4
  %cmp4 = icmp slt i32 %3, 3
  br i1 %cmp4, label %for.body5, label %for.end

for.body5:                                        ; preds = %for.cond3
  %4 = load i32, i32* @c, align 4
  %tobool = icmp ne i32 %4, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %for.body5
  %5 = load i32, i32* @a, align 4
  %dec = add nsw i32 %5, -1
  store i32 %dec, i32* @a, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body5
  %6 = load i32, i32* @e, align 4
  %idxprom6 = sext i32 %6 to i64
  %arrayidx7 = getelementptr inbounds [3 x i8], [3 x i8]* getelementptr inbounds ([1 x [3 x i8]], [1 x [3 x i8]]* @f, i32 0, i64 0), i32 0, i64 %idxprom6
  store i8 1, i8* %arrayidx7, align 1
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %7 = load i32, i32* @e, align 4
  %inc = add nsw i32 %7, 1
  store i32 %inc, i32* @e, align 4
  br label %for.cond3

for.end:                                          ; preds = %for.cond3
  br label %for.inc8

for.inc8:                                         ; preds = %for.end
  %8 = load i32, i32* @d, align 4
  %inc9 = add nsw i32 %8, 1
  store i32 %inc9, i32* @d, align 4
  br label %for.cond

for.end10:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: nounwind uwtable
define i32 @main() {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  call void @fn1()
  %0 = load i8, i8* getelementptr inbounds ([1 x [3 x i8]], [1 x [3 x i8]]* @f, i32 0, i64 0, i64 1), align 1
  %conv = sext i8 %0 to i32
  %cmp = icmp ne i32 %conv, 1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @abort()
  unreachable

if.end:                                           ; preds = %entry
  ret i32 0
}

; Function Attrs: noreturn nounwind
declare void @abort()
