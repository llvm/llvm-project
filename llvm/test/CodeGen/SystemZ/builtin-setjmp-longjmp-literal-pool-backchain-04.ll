; -mbackchain option
; Test output for inline Literal Pool with switch statement.
; len of malloc is global variable.
; RUN: clang -mbackchain -O2 -o %t %s
; RUN: %t | FileCheck %s

; ModuleID = 'builtin-setjmp-longjmp-literal-pool-04.c'
source_filename = "builtin-setjmp-longjmp-literal-pool-04.c"
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
@len = dso_local local_unnamed_addr global i32 10, align 4
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
; CHECK: First __builtin_setjmp in func1
; CHECK: Second __builtin_setjmp in func1
; CHECK: In func4
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
; CHECK: In func3
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

  %0 = load i32, ptr @len, align 4, !tbaa !4
  %conv = sext i32 %0 to i64
  %mul = shl nsw i64 %conv, 2
  %1 = alloca i8, i64 %mul, align 8
  %2 = tail call ptr asm sideeffect "larl $0, .LC101", "={r13}"() #4, !srcloc !8
  %add.ptr = getelementptr i8, ptr %2, i64 8
  %3 = load i32, ptr @len, align 4, !tbaa !4
  %cmp72 = icmp sgt i32 %3, 0
  br i1 %cmp72, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %4 = zext nneg i32 %3 to i64
  %5 = shl nuw nsw i64 %4, 2
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %1, ptr align 4 %add.ptr, i64 %5, i1 false), !tbaa !4
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body.preheader, %entry
  %6 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf2)
  %cmp3 = icmp eq i32 %6, 0
  br i1 %cmp3, label %if.then, label %if.else36

if.then:                                          ; preds = %for.cond.cleanup
  %puts69 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.15)
  %7 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf3)
  %cmp5 = icmp eq i32 %7, 0
  br i1 %cmp5, label %if.then7, label %if.else

if.then7:                                         ; preds = %if.then
  %puts71 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.17)
  tail call void @func4()
  unreachable

if.else:                                          ; preds = %if.then
  %puts70 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.16)
  %8 = load i32, ptr %add.ptr, align 4, !tbaa !4
  %call10 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %8)
  %9 = load i32, ptr @len, align 4, !tbaa !4
  %cmp1376 = icmp sgt i32 %9, 0
  br i1 %cmp1376, label %for.body16, label %for.cond.cleanup15

for.cond.cleanup15:                               ; preds = %for.body16, %if.else
  %10 = tail call ptr asm sideeffect "larl $0, .LC202", "={r13}"() #4, !srcloc !9
  tail call void @func3()
  unreachable

for.body16:                                       ; preds = %if.else, %for.body16
  %indvars.iv82 = phi i64 [ %indvars.iv.next83, %for.body16 ], [ 0, %if.else ]
  %arrayidx18 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv82
  %11 = load i32, ptr %arrayidx18, align 4, !tbaa !4
  %call19 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %11)
  %indvars.iv.next83 = add nuw nsw i64 %indvars.iv82, 1
  %12 = load i32, ptr @len, align 4, !tbaa !4
  %13 = sext i32 %12 to i64
  %cmp13 = icmp slt i64 %indvars.iv.next83, %13
  br i1 %cmp13, label %for.body16, label %for.cond.cleanup15, !llvm.loop !10

if.else36:                                        ; preds = %for.cond.cleanup
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.14)
  %14 = load i32, ptr %add.ptr, align 4, !tbaa !4
  %call38 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, i32 noundef signext %14)
  %15 = load i32, ptr @len, align 4, !tbaa !4
  %cmp4174 = icmp sgt i32 %15, 0
  br i1 %cmp4174, label %for.body44, label %for.cond.cleanup43

for.cond.cleanup43:                               ; preds = %for.body44, %if.else36
  tail call void @func2()
  unreachable

for.body44:                                       ; preds = %if.else36, %for.body44
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body44 ], [ 0, %if.else36 ]
  %arrayidx46 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv
  %16 = load i32, ptr %arrayidx46, align 4, !tbaa !4
  %call47 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %16)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %17 = load i32, ptr @len, align 4, !tbaa !4
  %18 = sext i32 %17 to i64
  %cmp41 = icmp slt i64 %indvars.iv.next, %18
  br i1 %cmp41, label %for.body44, label %for.cond.cleanup43, !llvm.loop !12
}

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #4

; Function Attrs: nounwind
define dso_local noundef signext i32 @main(i32 noundef signext %argc, ptr nocapture noundef readnone %argv) local_unnamed_addr #5 {
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

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #7

attributes #0 = { noinline noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { nofree nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #2 = { noreturn nounwind }
attributes #3 = { noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #4 = { nounwind }
attributes #5 = { nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #6 = { nofree nounwind }
attributes #7 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git b289df99d26b008287e18cdb0858bc569de3f2ad)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{i64 1677}
!9 = !{i64 2355}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
!12 = distinct !{!12, !11}
