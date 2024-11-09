; -mbackchain
; This tests Frame Pointer.
; This tests program output for Frame Pointer.
; Non-volatile local variable being modified between setjmp and longjmp call.
; This test is without optimization -O0, modified value persists.

; RUN: clang -mbackchain -O0 -o %t %s
; RUN: %t | FileCheck %s

; ModuleID = 'builtin-setjmp-longjmp-alloca-04.c'
source_filename = "builtin-setjmp-longjmp-alloca-04.c"
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
@.str.6 = private unnamed_addr constant [17 x i8] c"Dynamic var: %d\0A\00", align 2
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
; CHECK: Dynamic var: 10
; CHECK: Returned from func3
; CHECK: Dynamic var: 20

  %dynamic_var = alloca ptr, align 8
  %0 = alloca i8, i64 4, align 8
  store ptr %0, ptr %dynamic_var, align 8
  %1 = load ptr, ptr %dynamic_var, align 8
  store i32 10, ptr %1, align 4
  %2 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf2)
  %cmp = icmp eq i32 %2, 0
  br i1 %cmp, label %if.then, label %if.else6

if.then:                                          ; preds = %entry
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str.3)
  %3 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf3)
  %cmp1 = icmp eq i32 %3, 0
  br i1 %cmp1, label %if.then2, label %if.else

if.then2:                                         ; preds = %if.then
  %call3 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.4)
  call void @func4()
  br label %if.end

if.else:                                          ; preds = %if.then
  %call4 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.5)
  %4 = load ptr, ptr %dynamic_var, align 8
  %5 = load i32, ptr %4, align 4
  %call5 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6, i32 noundef signext %5)
  %6 = load ptr, ptr %dynamic_var, align 8
  store i32 20, ptr %6, align 4
  call void @func3()
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then2
  br label %if.end9

if.else6:                                         ; preds = %entry
  %call7 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.7)
  %7 = load ptr, ptr %dynamic_var, align 8
  %8 = load i32, ptr %7, align 4
  %call8 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.6, i32 noundef signext %8)
  call void @func2()
  br label %if.end9

if.end9:                                          ; preds = %if.else6, %if.end
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

attributes #0 = { noinline nounwind optnone "backchain" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { "backchain" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #2 = { noreturn nounwind }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git a0433728375e658551506ce43b0848200fdd6e61)"}
