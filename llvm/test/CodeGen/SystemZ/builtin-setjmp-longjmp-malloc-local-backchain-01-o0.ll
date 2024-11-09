; -mbackchain
; Non-volatile local malloc'd variable being modified between setjmp and longjmp call.
; This test is without optimization -O0, modified value persists.

; RUN: clang -mbackchain -O0 -o %t %s
; RUN: %t | FileCheck %s
; CHECK: setjmp has been called local_var=10
; CHECK: Calling function foo local_var=20
; CHECK: Calling longjmp from inside function foo
; CHECK: longjmp has been called local_val=20
; CHECK: Performing function recover

; ModuleID = 'builtin-setjmp-longjmp-malloc-local-01.c'
source_filename = "builtin-setjmp-longjmp-malloc-local-01.c"
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
  %local_var = alloca ptr, align 8
  %call = call noalias ptr @malloc(i64 noundef 4) #6
  store ptr %call, ptr %local_var, align 8
  %0 = load ptr, ptr %local_var, align 8
  store i32 10, ptr %0, align 4
  %1 = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  %cmp = icmp ne i32 %1, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %2 = load ptr, ptr %local_var, align 8
  %3 = load i32, ptr %2, align 4
  %call1 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.2, i32 noundef signext %3)
  call void @recover()
  call void @exit(i32 noundef signext 0) #2
  unreachable

if.end:                                           ; preds = %entry
  %4 = load ptr, ptr %local_var, align 8
  %5 = load i32, ptr %4, align 4
  %call2 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.3, i32 noundef signext %5)
  %6 = load ptr, ptr %local_var, align 8
  store i32 20, ptr %6, align 4
  %7 = load ptr, ptr %local_var, align 8
  %8 = load i32, ptr %7, align 4
  %call3 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.4, i32 noundef signext %8)
  call void @foo()
  %9 = load ptr, ptr %local_var, align 8
  %10 = load i32, ptr %9, align 4
  %call4 = call signext i32 (ptr, ...) @printf(ptr noundef @.str.5, i32 noundef signext %10)
  ret i32 0
}

; Function Attrs: nounwind allocsize(0)
declare noalias ptr @malloc(i64 noundef) #3

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #4

; Function Attrs: noreturn nounwind
declare void @exit(i32 noundef signext) #5

attributes #0 = { noinline nounwind optnone "backchain" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { "backchain" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #2 = { noreturn nounwind }
attributes #3 = { nounwind allocsize(0) "backchain" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #4 = { nounwind }
attributes #5 = { noreturn nounwind "backchain" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #6 = { nounwind allocsize(0) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git a0433728375e658551506ce43b0848200fdd6e61)"}
