; Test the output with  -mbackchain option.

; The behavior of local variables between setjmp and longjmp in C is undefined.
; This means that the C standard does not guarantee what will happen
; to their values.

; Output from gcc at all optimization levels or clang -O0 is:

;First __builtin_setjmp in func1
;Second __builtin_setjmp in func1
;Returned from func4
;value_ptr : 954219
;Returned from func3
;value_ptr: 420420

; RUN: clang -O2 -mbackchain -o %t %s
; RUN: %t | FileCheck %s

; ModuleID = 'builtin-setjmp-longjmp-literal-pool-01.c'
source_filename = "builtin-setjmp-longjmp-literal-pool-01.c"
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

module asm ".LC101:"
module asm ".long 123454"
module asm ".long 451233"
module asm ".long 954219"
module asm ".long 466232"
module asm ".long 955551"
module asm ".long 687823"
module asm ".long 555123"
module asm ".long 777723"
module asm ".long 985473"
module asm ".long 190346"
module asm ".long 420420"
module asm ".long 732972"
module asm ".long 971166"
module asm ".long 123454"
module asm ".long 451233"
module asm ".long 954219"
module asm ".long 466232"
module asm ".long 955551"
module asm ".long 687823"
module asm ".long 555123"

@buf3 = dso_local global [10 x ptr] zeroinitializer, align 8
@buf2 = dso_local global [10 x ptr] zeroinitializer, align 8
@buf1 = dso_local global [10 x ptr] zeroinitializer, align 8
@.str.6 = private unnamed_addr constant [16 x i8] c"value_ptr : %d\0A\00", align 2
@.str.8 = private unnamed_addr constant [15 x i8] c"value_ptr: %d\0A\00", align 2
@str = private unnamed_addr constant [9 x i8] c"In func4\00", align 1
@str.11 = private unnamed_addr constant [9 x i8] c"In func3\00", align 1
@str.12 = private unnamed_addr constant [9 x i8] c"In func2\00", align 1
@str.13 = private unnamed_addr constant [20 x i8] c"Returned from func3\00", align 1
@str.14 = private unnamed_addr constant [32 x i8] c"First __builtin_setjmp in func1\00", align 1
@str.15 = private unnamed_addr constant [20 x i8] c"Returned from func4\00", align 1
@str.16 = private unnamed_addr constant [33 x i8] c"Second __builtin_setjmp in func1\00", align 1
@str.17 = private unnamed_addr constant [44 x i8] c"In main, after __builtin_longjmp from func1\00", align 1
@str.18 = private unnamed_addr constant [20 x i8] c"In main, first time\00", align 1

; Function Attrs: noinline noreturn nounwind
define dso_local void @func4() local_unnamed_addr #0 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf3)
  unreachable
}

; Function Attrs: nofree nounwind
declare noundef signext i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: noreturn nounwind
declare void @llvm.eh.sjlj.longjmp(ptr) #2

; Function Attrs: noinline noreturn nounwind
define dso_local void @func3() local_unnamed_addr #0 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.11)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf2)
  unreachable
}

; Function Attrs: noinline noreturn nounwind
define dso_local void @func2() local_unnamed_addr #0 {
entry:
; CHECK: First __builtin_setjmp in func1
; CHECK: Second __builtin_setjmp in func1
; CHECK: Returned from func4
; CHECK: value_ptr : 954219
; CHECK: Returned from func3
; CHECK: value_ptr: 954219

  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.12)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf1)
  unreachable
}

; Function Attrs: noreturn nounwind
define dso_local noundef signext i32 @func1() local_unnamed_addr #3 {
entry:
  %0 = tail call ptr asm sideeffect "larl $0, .LC101", "={r13}"() #4, !srcloc !4
  %add.ptr = getelementptr inbounds i8, ptr %0, i64 8
  %1 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf2)
  %cmp = icmp eq i32 %1, 0
  br i1 %cmp, label %if.then, label %if.else7

if.then:                                          ; preds = %entry
  %puts13 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.14)
  %2 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf3)
  %cmp1 = icmp eq i32 %2, 0
  br i1 %cmp1, label %if.then2, label %if.else

if.then2:                                         ; preds = %if.then
  %puts15 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.16)
  tail call void @func4()
  unreachable

if.else:                                          ; preds = %if.then
  %puts14 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.15)
  %3 = load i32, ptr %add.ptr, align 4, !tbaa !5
  %call5 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %3)
  tail call void @func3()
  unreachable

if.else7:                                         ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.13)
  %4 = load i32, ptr %add.ptr, align 4, !tbaa !5
  %call9 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef signext %4)
  tail call void @func2()
  unreachable
}

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #4

; Function Attrs: nounwind
define dso_local noundef signext i32 @main() local_unnamed_addr #5 {
entry:
  %0 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf1)
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %puts3 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.18)
  %call1 = tail call signext i32 @func1()
  unreachable

if.else:                                          ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.17)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #6

attributes #0 = { noinline noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { nofree nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #2 = { noreturn nounwind }
attributes #3 = { noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #4 = { nounwind }
attributes #5 = { nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #6 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git b289df99d26b008287e18cdb0858bc569de3f2ad)"}
!4 = !{i64 1166}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
