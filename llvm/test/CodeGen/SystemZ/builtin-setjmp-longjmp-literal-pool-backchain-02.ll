; Test the output with -mbackchain option.

; The behavior of local variables between setjmp and longjmp in C is undefined.
; This means that the C standard does not guarantee what will happen
; to their values.

; Output from gcc at all optimization levels or clang -O0 is:

;First __builtin_setjmp in func1
;Second __builtin_setjmp in func1
;Returned from func4
;value_ptr : 954219
;arr: 954219
;arr: 466232
;arr: 955551
;arr: 687823
;arr: 555123
;arr: 777723
;arr: 985473
;arr: 190346
;arr: 420420
;arr: 732972
;Returned from func3
;value_ptr: 971166
;arr: 971166
;arr: 123454
;arr: 451233
;arr: 954219
;arr: 466232
;arr: 955551
;arr: 687823
;arr: 555123
;arr: 123454
;arr: 451233

; RUN: clang -mbackchain -O2 -o %t %s
; RUN: %t | FileCheck %s

; ModuleID = 'builtin-setjmp-longjmp-literal-pool-02.c'
source_filename = "builtin-setjmp-longjmp-literal-pool-02.c"
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
module asm ".LC202:"
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

@buf3 = dso_local global [10 x ptr] zeroinitializer, align 8
@buf2 = dso_local global [10 x ptr] zeroinitializer, align 8
@buf1 = dso_local global [10 x ptr] zeroinitializer, align 8
@.str.6 = private unnamed_addr constant [16 x i8] c"value_ptr : %d\0A\00", align 2
@.str.7 = private unnamed_addr constant [9 x i8] c"arr: %d\0A\00", align 2
@.str.9 = private unnamed_addr constant [15 x i8] c"value_ptr: %d\0A\00", align 2
@str = private unnamed_addr constant [9 x i8] c"In func4\00", align 1
@str.12 = private unnamed_addr constant [9 x i8] c"In func3\00", align 1
@str.13 = private unnamed_addr constant [9 x i8] c"In func2\00", align 1
@str.14 = private unnamed_addr constant [20 x i8] c"Returned from func3\00", align 1
@str.15 = private unnamed_addr constant [32 x i8] c"First __builtin_setjmp in func1\00", align 1
@str.16 = private unnamed_addr constant [20 x i8] c"Returned from func4\00", align 1
@str.17 = private unnamed_addr constant [33 x i8] c"Second __builtin_setjmp in func1\00", align 1
@str.18 = private unnamed_addr constant [44 x i8] c"In main, after __builtin_longjmp from func1\00", align 1
@str.19 = private unnamed_addr constant [20 x i8] c"In main, first time\00", align 1

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
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.12)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf2)
  unreachable
}

; Function Attrs: noinline noreturn nounwind
define dso_local void @func2() local_unnamed_addr #0 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.13)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf1)
  unreachable
}

; Function Attrs: noreturn nounwind
define dso_local noundef signext i32 @func1() local_unnamed_addr #3 {
entry:

; CHECK: Returned from func4
; CHECK: value_ptr : 954219
; CHECK: arr: 954219
; CHECK: arr: 466232
; CHECK: arr: 955551
; CHECK: arr: 687823
; CHECK: arr: 555123
; CHECK: arr: 777723
; CHECK: arr: 985473
; CHECK: arr: 190346
; CHECK: arr: 420420
; CHECK: arr: 732972
; CHECK: Returned from func3
; CHECK: value_ptr: 954219
; CHECK: arr: 954219
; CHECK: arr: 466232
; CHECK: arr: 955551
; CHECK: arr: 687823
; CHECK: arr: 555123
; CHECK: arr: 777723
; CHECK: arr: 985473
; CHECK: arr: 190346
; CHECK: arr: 420420
; CHECK: arr: 732972

  %0 = tail call ptr asm sideeffect "larl $0, .LC101", "={r13}"() #4, !srcloc !4
  %add.ptr = getelementptr inbounds i8, ptr %0, i64 8
  %.sroa.0.0.copyload = load i32, ptr %add.ptr, align 4, !tbaa !5
  %.sroa.4.0.add.ptr.sroa_idx = getelementptr inbounds i8, ptr %0, i64 12
  %.sroa.4.0.copyload = load i32, ptr %.sroa.4.0.add.ptr.sroa_idx, align 4, !tbaa !5
  %.sroa.6.0.add.ptr.sroa_idx = getelementptr inbounds i8, ptr %0, i64 16
  %.sroa.6.0.copyload = load i32, ptr %.sroa.6.0.add.ptr.sroa_idx, align 4, !tbaa !5
  %.sroa.8.0.add.ptr.sroa_idx = getelementptr inbounds i8, ptr %0, i64 20
  %.sroa.8.0.copyload = load i32, ptr %.sroa.8.0.add.ptr.sroa_idx, align 4, !tbaa !5
  %.sroa.10.0.add.ptr.sroa_idx = getelementptr inbounds i8, ptr %0, i64 24
  %.sroa.10.0.copyload = load i32, ptr %.sroa.10.0.add.ptr.sroa_idx, align 4, !tbaa !5
  %.sroa.12.0.add.ptr.sroa_idx = getelementptr inbounds i8, ptr %0, i64 28
  %.sroa.12.0.copyload = load i32, ptr %.sroa.12.0.add.ptr.sroa_idx, align 4, !tbaa !5
  %.sroa.14.0.add.ptr.sroa_idx = getelementptr inbounds i8, ptr %0, i64 32
  %.sroa.14.0.copyload = load i32, ptr %.sroa.14.0.add.ptr.sroa_idx, align 4, !tbaa !5
  %.sroa.16.0.add.ptr.sroa_idx = getelementptr inbounds i8, ptr %0, i64 36
  %.sroa.16.0.copyload = load i32, ptr %.sroa.16.0.add.ptr.sroa_idx, align 4, !tbaa !5
  %.sroa.18.0.add.ptr.sroa_idx = getelementptr inbounds i8, ptr %0, i64 40
  %.sroa.18.0.copyload = load i32, ptr %.sroa.18.0.add.ptr.sroa_idx, align 4, !tbaa !5
  %.sroa.20.0.add.ptr.sroa_idx = getelementptr inbounds i8, ptr %0, i64 44
  %.sroa.20.0.copyload = load i32, ptr %.sroa.20.0.add.ptr.sroa_idx, align 4, !tbaa !5
  %1 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf2)
  %cmp2 = icmp eq i32 %1, 0
  br i1 %cmp2, label %if.then, label %if.else32

if.then:                                          ; preds = %entry
  %puts64 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.15)
  %2 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf3)
  %cmp3 = icmp eq i32 %2, 0
  br i1 %cmp3, label %if.then4, label %if.else

if.then4:                                         ; preds = %if.then
  %puts66 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.17)
  tail call void @func4()
  unreachable

if.else:                                          ; preds = %if.then
  %puts65 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.16)
  %3 = load i32, ptr %add.ptr, align 4, !tbaa !5
  %call7 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %3)
  %call15 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.0.0.copyload)
  %call15.1 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.4.0.copyload)
  %call15.2 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.6.0.copyload)
  %call15.3 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.8.0.copyload)
  %call15.4 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.10.0.copyload)
  %call15.5 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.12.0.copyload)
  %call15.6 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.14.0.copyload)
  %call15.7 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.16.0.copyload)
  %call15.8 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.18.0.copyload)
  %call15.9 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.20.0.copyload)
  %4 = tail call ptr asm sideeffect "larl $0, .LC202", "={r13}"() #4, !srcloc !9
  tail call void @func3()
  unreachable

if.else32:                                        ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.14)
  %5 = load i32, ptr %add.ptr, align 4, !tbaa !5
  %call34 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, i32 noundef signext %5)
  %call42 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.0.0.copyload)
  %call42.1 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.4.0.copyload)
  %call42.2 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.6.0.copyload)
  %call42.3 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.8.0.copyload)
  %call42.4 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.10.0.copyload)
  %call42.5 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.12.0.copyload)
  %call42.6 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.14.0.copyload)
  %call42.7 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.16.0.copyload)
  %call42.8 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.18.0.copyload)
  %call42.9 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %.sroa.20.0.copyload)
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
  %puts3 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.19)
  %call1 = tail call signext i32 @func1()
  unreachable

if.else:                                          ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.18)
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
!4 = !{i64 1661}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{i64 2337}
