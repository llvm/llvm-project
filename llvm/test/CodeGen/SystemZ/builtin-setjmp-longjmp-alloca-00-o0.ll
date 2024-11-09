; This tests program output for Frame Pointer.
; Non-volatile local variable being modified between setjmp and longjmp call.
; This test is without optimization -O0, modified value persists.

; RUN: clang -O0 -o %t %s
; RUN: %t | FileCheck %s

; ModuleID = 'builtin-setjmp-longjmp-alloca-00.c'
source_filename = "builtin-setjmp-longjmp-alloca-00.c"
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

@.str = private unnamed_addr constant [10 x i8] c"In func4\0A\00", align 2
@buf3 = dso_local global [10 x ptr] zeroinitializer, align 8
@.str.1 = private unnamed_addr constant [10 x i8] c"In func3\0A\00", align 2
@buf2 = dso_local global [10 x ptr] zeroinitializer, align 8
@.str.2 = private unnamed_addr constant [10 x i8] c"In func2\0A\00", align 2
@buf1 = dso_local global [10 x ptr] zeroinitializer, align 8
@.str.3 = private unnamed_addr constant [33 x i8] c"First __builtin_setjmp in func1\0A\00", align 2
@.str.4 = private unnamed_addr constant [34 x i8] c"Second __builtin_setjmp in func1\0A\00", align 2
@.str.5 = private unnamed_addr constant [21 x i8] c"Returned from func4\0A\00", align 2
@.str.6 = private unnamed_addr constant [9 x i8] c"arr: %d\0A\00", align 2
@.str.7 = private unnamed_addr constant [21 x i8] c"Returned from func3\0A\00", align 2
@.str.8 = private unnamed_addr constant [21 x i8] c"In main, first time\0A\00", align 2
@.str.9 = private unnamed_addr constant [45 x i8] c"In main, after __builtin_longjmp from func1\0A\00", align 2

; Function Attrs: noinline nounwind optnone
define dso_local void @func4() #0 {
entry:
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str)
  call void @llvm.eh.sjlj.longjmp(ptr @buf3)
  unreachable
}

declare signext i32 @printf(ptr noundef, ...) #1

; Function Attrs: noreturn nounwind
declare void @llvm.eh.sjlj.longjmp(ptr) #2

; Function Attrs: noinline nounwind optnone
define dso_local void @func3() #0 {
entry:
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1)
  call void @llvm.eh.sjlj.longjmp(ptr @buf2)
  unreachable
}

; Function Attrs: noinline nounwind optnone
define dso_local void @func2() #0 {
entry:
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2)
  call void @llvm.eh.sjlj.longjmp(ptr @buf1)
  unreachable
}

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @func1() #0 {
entry:
; CHECK: First __builtin_setjmp in func1
; CHECK: Second __builtin_setjmp in func1
; CHECK: Returned from func4
; CHECK: arr: 0
; CHECK: arr: 2
; CHECK: arr: 6
; CHECK: arr: 12
; CHECK: arr: 20
; CHECK: arr: 30
; CHECK: arr: 42
; CHECK: arr: 56
; CHECK: arr: 72
; CHECK: arr: 90
; CHECK: Returned from func3
; CHECK: arr: 0
; CHECK: arr: 3 
; CHECK: arr: 14 
; CHECK: arr: 39 
; CHECK: arr: 84 
; CHECK: arr: 155 
; CHECK: arr: 258 
; CHECK: arr: 399 
; CHECK: arr: 584 
; CHECK: arr: 819 

  %len = alloca i32, align 4
  %arr = alloca ptr, align 8
  %i = alloca i32, align 4
  %i10 = alloca i32, align 4
  %i21 = alloca i32, align 4
  %i38 = alloca i32, align 4
  store i32 10, ptr %len, align 4
  %0 = load i32, ptr %len, align 4
  %conv = sext i32 %0 to i64
  %mul = mul i64 %conv, 4
  %1 = alloca i8, i64 %mul, align 8
  store ptr %1, ptr %arr, align 8
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, ptr %i, align 4
  %3 = load i32, ptr %len, align 4
  %cmp = icmp slt i32 %2, %3
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %4 = load i32, ptr %i, align 4
  %5 = load i32, ptr %i, align 4
  %mul2 = mul nsw i32 %4, %5
  %6 = load i32, ptr %i, align 4
  %add = add nsw i32 %mul2, %6
  %7 = load ptr, ptr %arr, align 8
  %8 = load i32, ptr %i, align 4
  %idxprom = sext i32 %8 to i64
  %arrayidx = getelementptr inbounds i32, ptr %7, i64 %idxprom
  store i32 %add, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %9 = load i32, ptr %i, align 4
  %inc = add nsw i32 %9, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond, !llvm.loop !5

for.end:                                          ; preds = %for.cond
  %10 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf2)
  %cmp3 = icmp eq i32 %10, 0
  br i1 %cmp3, label %if.then, label %if.else36

if.then:                                          ; preds = %for.end
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str.3)
  %11 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf3)
  %cmp5 = icmp eq i32 %11, 0
  br i1 %cmp5, label %if.then7, label %if.else

if.then7:                                         ; preds = %if.then
  %call8 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.4)
  call void @func4()
  br label %if.end

if.else:                                          ; preds = %if.then
  %call9 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.5)
  store i32 0, ptr %i10, align 4
  br label %for.cond11

for.cond11:                                       ; preds = %for.inc18, %if.else
  %12 = load i32, ptr %i10, align 4
  %13 = load i32, ptr %len, align 4
  %cmp12 = icmp slt i32 %12, %13
  br i1 %cmp12, label %for.body14, label %for.end20

for.body14:                                       ; preds = %for.cond11
  %14 = load ptr, ptr %arr, align 8
  %15 = load i32, ptr %i10, align 4
  %idxprom15 = sext i32 %15 to i64
  %arrayidx16 = getelementptr inbounds i32, ptr %14, i64 %idxprom15
  %16 = load i32, ptr %arrayidx16, align 4
  %call17 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6, i32 noundef signext %16)
  br label %for.inc18

for.inc18:                                        ; preds = %for.body14
  %17 = load i32, ptr %i10, align 4
  %inc19 = add nsw i32 %17, 1
  store i32 %inc19, ptr %i10, align 4
  br label %for.cond11, !llvm.loop !7

for.end20:                                        ; preds = %for.cond11
  store i32 0, ptr %i21, align 4
  br label %for.cond22

for.cond22:                                       ; preds = %for.inc33, %for.end20
  %18 = load i32, ptr %i21, align 4
  %19 = load i32, ptr %len, align 4
  %cmp23 = icmp slt i32 %18, %19
  br i1 %cmp23, label %for.body25, label %for.end35

for.body25:                                       ; preds = %for.cond22
  %20 = load i32, ptr %i21, align 4
  %21 = load i32, ptr %i21, align 4
  %mul26 = mul nsw i32 %20, %21
  %22 = load i32, ptr %i21, align 4
  %mul27 = mul nsw i32 %mul26, %22
  %23 = load i32, ptr %i21, align 4
  %24 = load i32, ptr %i21, align 4
  %mul28 = mul nsw i32 %23, %24
  %add29 = add nsw i32 %mul27, %mul28
  %25 = load i32, ptr %i21, align 4
  %add30 = add nsw i32 %add29, %25
  %26 = load ptr, ptr %arr, align 8
  %27 = load i32, ptr %i21, align 4
  %idxprom31 = sext i32 %27 to i64
  %arrayidx32 = getelementptr inbounds i32, ptr %26, i64 %idxprom31
  store i32 %add30, ptr %arrayidx32, align 4
  br label %for.inc33

for.inc33:                                        ; preds = %for.body25
  %28 = load i32, ptr %i21, align 4
  %inc34 = add nsw i32 %28, 1
  store i32 %inc34, ptr %i21, align 4
  br label %for.cond22, !llvm.loop !8

for.end35:                                        ; preds = %for.cond22
  call void @func3()
  br label %if.end

if.end:                                           ; preds = %for.end35, %if.then7
  br label %if.end49

if.else36:                                        ; preds = %for.end
  %call37 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.7)
  store i32 0, ptr %i38, align 4
  br label %for.cond39

for.cond39:                                       ; preds = %for.inc46, %if.else36
  %29 = load i32, ptr %i38, align 4
  %30 = load i32, ptr %len, align 4
  %cmp40 = icmp slt i32 %29, %30
  br i1 %cmp40, label %for.body42, label %for.end48

for.body42:                                       ; preds = %for.cond39
  %31 = load ptr, ptr %arr, align 8
  %32 = load i32, ptr %i38, align 4
  %idxprom43 = sext i32 %32 to i64
  %arrayidx44 = getelementptr inbounds i32, ptr %31, i64 %idxprom43
  %33 = load i32, ptr %arrayidx44, align 4
  %call45 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6, i32 noundef signext %33)
  br label %for.inc46

for.inc46:                                        ; preds = %for.body42
  %34 = load i32, ptr %i38, align 4
  %inc47 = add nsw i32 %34, 1
  store i32 %inc47, ptr %i38, align 4
  br label %for.cond39, !llvm.loop !9

for.end48:                                        ; preds = %for.cond39
  call void @func2()
  br label %if.end49

if.end49:                                         ; preds = %for.end48, %if.end
  ret i32 0
}

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #3

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  %0 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf1)
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str.8)
  %call1 = call signext i32 @func1()
  br label %if.end

if.else:                                          ; preds = %entry
  %call2 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.9)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret i32 0
}

attributes #0 = { noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #2 = { noreturn nounwind }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git a0433728375e658551506ce43b0848200fdd6e61)"}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.mustprogress"}
!7 = distinct !{!7, !6}
!8 = distinct !{!8, !6}
!9 = distinct !{!9, !6}
