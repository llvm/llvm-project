; global malloc'd variable being modified between setjmp and longjmp call.
; modified value persists.

; RUN: clang -O2 -o %t %s
; RUN: %t | FileCheck %s

; CHECK: setjmp has been called global_var=10
; CHECK: Calling function foo global_var=20
; CHECK: Calling longjmp from inside function foo
; CHECK: longjmp has been called local_val=20
; CHECK: Performing function recover

; ModuleID = 'builtin-setjmp-longjmp-malloc-global-0l.c'
source_filename = "builtin-setjmp-longjmp-malloc-global-0l.c"
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

@buf = dso_local global [20 x ptr] zeroinitializer, align 8
@global_var = dso_local local_unnamed_addr global ptr null, align 8
@.str.2 = private unnamed_addr constant [39 x i8] c"longjmp has been called local_val=%d \0A\00", align 2
@.str.3 = private unnamed_addr constant [39 x i8] c"setjmp has been called global_var=%d \0A\00", align 2
@.str.4 = private unnamed_addr constant [37 x i8] c"Calling function foo global_var=%d \0A\00", align 2
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
  store ptr %call, ptr @global_var, align 8, !tbaa !4
  store i32 10, ptr %call, align 4, !tbaa !8
  %0 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf)
  %cmp.not = icmp eq i32 %0, 0
  %1 = load ptr, ptr @global_var, align 8, !tbaa !4
  %2 = load i32, ptr %1, align 4, !tbaa !8
  br i1 %cmp.not, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %call1 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef signext %2)
  tail call void @recover()
  tail call void @exit(i32 noundef signext 0) #2
  unreachable

if.end:                                           ; preds = %entry
  %call2 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef signext %2)
  %3 = load ptr, ptr @global_var, align 8, !tbaa !4
  store i32 20, ptr %3, align 4, !tbaa !8
  %call3 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef signext 20)
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
!5 = !{!"any pointer", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !6, i64 0}
