; volatile local malloc'd variable being modified between setjmp and longjmp call.
; modified value persists.

; RUN: clang -O2 -o %t %s
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

; ModuleID = 'builtin-setjmp-longjmp-malloc-volatile-02.c'
source_filename = "builtin-setjmp-longjmp-malloc-volatile-02.c"
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

@buf = dso_local global [20 x ptr] zeroinitializer, align 8
@.str.2 = private unnamed_addr constant [39 x i8] c"longjmp has been called local_val=%d \0A\00", align 2
@.str.3 = private unnamed_addr constant [38 x i8] c"setjmp has been called local_var=%d \0A\00", align 2
@.str.4 = private unnamed_addr constant [36 x i8] c"Calling function foo local_var=%d \0A\00", align 2
@str = private unnamed_addr constant [41 x i8] c"Calling longjmp from inside function foo\00", align 1
@str.6 = private unnamed_addr constant [28 x i8] c"Performing function recover\00", align 1

; Function Attrs: noinline noreturn nounwind
define dso_local void @foo() local_unnamed_addr #0 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf)
  unreachable
}

; Function Attrs: nofree nounwind
declare noundef signext i32 @printf(ptr nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: noreturn nounwind
declare void @llvm.eh.sjlj.longjmp(ptr) #2

; Function Attrs: nofree noinline nounwind
define dso_local void @recover() local_unnamed_addr #3 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.6)
  ret void
}

; Function Attrs: noreturn nounwind
define dso_local noundef signext i32 @main() local_unnamed_addr #4 {
entry:
  %call = tail call noalias dereferenceable_or_null(4) ptr @malloc(i64 noundef 4) #9
  store volatile i32 10, ptr %call, align 4, !tbaa !4
  %0 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf)
  %cmp.not = icmp eq i32 %0, 0
  br i1 %cmp.not, label %if.end4, label %if.then

if.then:                                          ; preds = %entry
  %1 = load volatile i32, ptr %call, align 4, !tbaa !4
  %call1 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef signext %1)
  tail call void @recover()
  %2 = load volatile i32, ptr %call, align 4, !tbaa !4
  %cmp2.not = icmp eq i32 %2, 20
  br i1 %cmp2.not, label %if.end, label %if.then3

if.then3:                                         ; preds = %if.then
  tail call void @exit(i32 noundef signext 0) #2
  unreachable

if.end:                                           ; preds = %if.then
  store volatile i32 30, ptr %call, align 4, !tbaa !4
  br label %if.end4

if.end4:                                          ; preds = %if.end, %entry
  %3 = load volatile i32, ptr %call, align 4, !tbaa !4
  %call5 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef signext %3)
  %4 = load volatile i32, ptr %call, align 4, !tbaa !4
  %cmp6 = icmp eq i32 %4, 10
  %. = select i1 %cmp6, i32 20, i32 40
  store volatile i32 %., ptr %call, align 4, !tbaa !4
  %5 = load volatile i32, ptr %call, align 4, !tbaa !4
  %call9 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef signext %5)
  tail call void @foo()
  unreachable
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #5

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #6

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef signext) local_unnamed_addr #7

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #8

attributes #0 = { noinline noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #2 = { noreturn nounwind }
attributes #3 = { nofree noinline nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #4 = { noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #5 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #6 = { nounwind }
attributes #7 = { nofree noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #8 = { nofree nounwind }
attributes #9 = { nounwind allocsize(0) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git a0433728375e658551506ce43b0848200fdd6e61)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
