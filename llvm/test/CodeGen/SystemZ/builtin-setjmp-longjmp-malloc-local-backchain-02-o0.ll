; -mbackchain
; Non-volatile local malloc'd variable being modified between setjmp and longjmp call.
; This test is without optimization -O0, modified value persists.
; FIXME: Add this test case when local malloc'd non-volatile variable is fixed.
; It will create infinite loop.

; RUN: clang -O0 -o %t %s
; RUN: %t | FileCheck %s

; CHECK: setjmp has been called local_var=10
; CHECK: Calling function foo local_var=20
; CHECK: Calling longjmp from inside function foo
; CHECK: longjmp has been called local_val=20
; CHECK: Performing function recover
; CHECK: setjmp has been called local_var=30
; CHECK: Calling function foo local_var=40
; CHECK: Calling longjmp from inside function foo
; CHECK: longjmp has been called local_val=40
; CHECK: Performing function recover

; ModuleID = 'builtin-setjmp-longjmp-malloc-local-02.c'
source_filename = "builtin-setjmp-longjmp-malloc-local-02.c"
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

@.str = private unnamed_addr constant [42 x i8] c"Calling longjmp from inside function foo\0A\00", align 2
@buf = dso_local global [20 x ptr] zeroinitializer, align 8
@.str.1 = private unnamed_addr constant [29 x i8] c"Performing function recover\0A\00", align 2
@.str.2 = private unnamed_addr constant [39 x i8] c"longjmp has been called local_val=%d \0A\00", align 2
@.str.3 = private unnamed_addr constant [38 x i8] c"setjmp has been called local_var=%d \0A\00", align 2
@.str.4 = private unnamed_addr constant [36 x i8] c"Calling function foo local_var=%d \0A\00", align 2
@.str.5 = private unnamed_addr constant [50 x i8] c"This point should never be reached local_var=%d \0A\00", align 2

; Function Attrs: noinline nounwind optnone
define dso_local void @foo() #0 {
entry:
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str)
  call void @llvm.eh.sjlj.longjmp(ptr @buf)
  unreachable
}

declare signext i32 @printf(ptr noundef, ...) #1

; Function Attrs: noreturn nounwind
declare void @llvm.eh.sjlj.longjmp(ptr) #2

; Function Attrs: noinline nounwind optnone
define dso_local void @recover() #0 {
entry:
  %call = call signext i32 (ptr, ...) @printf(ptr noundef @.str.1)
  ret void
}

; Function Attrs: noinline nounwind optnone
define dso_local signext i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %local_var = alloca ptr, align 8
  store i32 0, ptr %retval, align 4
  %call = call noalias ptr @malloc(i64 noundef 4) #7
  store ptr %call, ptr %local_var, align 8
  %0 = load ptr, ptr %local_var, align 8
  store i32 10, ptr %0, align 4
  %1 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  %cmp = icmp ne i32 %1, 0
  br i1 %cmp, label %if.then, label %if.end4

if.then:                                          ; preds = %entry
  %2 = load ptr, ptr %local_var, align 8
  %3 = load i32, ptr %2, align 4
  %call1 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2, i32 noundef signext %3)
  call void @recover()
  %4 = load ptr, ptr %local_var, align 8
  %5 = load i32, ptr %4, align 4
  %cmp2 = icmp ne i32 %5, 20
  br i1 %cmp2, label %if.then3, label %if.end

if.then3:                                         ; preds = %if.then
  %6 = load ptr, ptr %local_var, align 8
  call void @free(ptr noundef %6) #4
  call void @exit(i32 noundef signext 0) #2
  unreachable

if.end:                                           ; preds = %if.then
  %7 = load ptr, ptr %local_var, align 8
  store i32 30, ptr %7, align 4
  br label %if.end4

if.end4:                                          ; preds = %if.end, %entry
  %8 = load ptr, ptr %local_var, align 8
  %9 = load i32, ptr %8, align 4
  %call5 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.3, i32 noundef signext %9)
  %10 = load ptr, ptr %local_var, align 8
  %11 = load i32, ptr %10, align 4
  %cmp6 = icmp eq i32 %11, 10
  br i1 %cmp6, label %if.then7, label %if.else

if.then7:                                         ; preds = %if.end4
  %12 = load ptr, ptr %local_var, align 8
  store i32 20, ptr %12, align 4
  br label %if.end8

if.else:                                          ; preds = %if.end4
  %13 = load ptr, ptr %local_var, align 8
  store i32 40, ptr %13, align 4
  br label %if.end8

if.end8:                                          ; preds = %if.else, %if.then7
  %14 = load ptr, ptr %local_var, align 8
  %15 = load i32, ptr %14, align 4
  %call9 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.4, i32 noundef signext %15)
  call void @foo()
  %16 = load ptr, ptr %local_var, align 8
  %17 = load i32, ptr %16, align 4
  %call10 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.5, i32 noundef signext %17)
  ret i32 0
}

; Function Attrs: nounwind allocsize(0)
declare noalias ptr @malloc(i64 noundef) #3

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #4

; Function Attrs: nounwind
declare void @free(ptr noundef) #5

; Function Attrs: noreturn nounwind
declare void @exit(i32 noundef signext) #6

attributes #0 = { noinline nounwind optnone "backchain" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { "backchain" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #2 = { noreturn nounwind }
attributes #3 = { nounwind allocsize(0) "backchain" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #4 = { nounwind }
attributes #5 = { nounwind "backchain" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #6 = { noreturn nounwind "backchain" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #7 = { nounwind allocsize(0) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git a0433728375e658551506ce43b0848200fdd6e61)"}
