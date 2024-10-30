; -mbackchain option 
; Test output for inline Literal Pool with switch statement.
; len of malloc is passed as argument.
; RUN: clang -mbackchain -O2 -o %t %s
; RUN: %t 10 | FileCheck %s

; ModuleID = 'builtin-setjmp-longjmp-literal-pool-03.c'
source_filename = "builtin-setjmp-longjmp-literal-pool-03.c"
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
@str.13 = private unnamed_addr constant [9 x i8] c"In func3\00", align 1
@str.14 = private unnamed_addr constant [9 x i8] c"In func2\00", align 1
@str.15 = private unnamed_addr constant [20 x i8] c"Returned from func3\00", align 1
@str.16 = private unnamed_addr constant [32 x i8] c"First __builtin_setjmp in func1\00", align 1
@str.17 = private unnamed_addr constant [20 x i8] c"Returned from func4\00", align 1
@str.18 = private unnamed_addr constant [33 x i8] c"Second __builtin_setjmp in func1\00", align 1
@str.19 = private unnamed_addr constant [44 x i8] c"In main, after __builtin_longjmp from func1\00", align 1
@str.20 = private unnamed_addr constant [20 x i8] c"In main, first time\00", align 1
@str.21 = private unnamed_addr constant [30 x i8] c"Usage: program_name <length> \00", align 1

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
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.13)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf2)
  unreachable
}

; Function Attrs: noinline noreturn nounwind
define dso_local void @func2() local_unnamed_addr #0 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.14)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf1)
  unreachable
}

; Function Attrs: noreturn nounwind
define dso_local noundef signext i32 @func1(i32 noundef signext %len) local_unnamed_addr #3 {
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

  %conv = sext i32 %len to i64
  %mul = shl nsw i64 %conv, 2
  %0 = alloca i8, i64 %mul, align 8
  %1 = tail call ptr asm sideeffect "larl $0, .LC101", "={r13}"() #4, !srcloc !4
  %add.ptr = getelementptr i8, ptr %1, i64 8
  %cmp76 = icmp sgt i32 %len, 0
  br i1 %cmp76, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %2 = zext nneg i32 %len to i64
  %3 = shl nuw nsw i64 %2, 2
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 8 %0, ptr align 4 %add.ptr, i64 %3, i1 false), !tbaa !5
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body.preheader, %entry
  %4 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf2)
  %cmp3 = icmp eq i32 %4, 0
  br i1 %cmp3, label %if.then, label %if.else36

if.then:                                          ; preds = %for.cond.cleanup
  %puts73 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.16)
  %5 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf3)
  %cmp5 = icmp eq i32 %5, 0
  br i1 %cmp5, label %if.then7, label %if.else

if.then7:                                         ; preds = %if.then
  %puts75 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.18)
  tail call void @func4()
  unreachable

if.else:                                          ; preds = %if.then
  %puts74 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.17)
  %6 = load i32, ptr %add.ptr, align 4, !tbaa !5
  %call10 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %6)
  br i1 %cmp76, label %for.body16.preheader, label %for.cond.cleanup15.thread

for.cond.cleanup15.thread:                        ; preds = %if.else
  %7 = tail call ptr asm sideeffect "larl $0, .LC202", "={r13}"() #4, !srcloc !9
  br label %for.cond.cleanup27

for.body16.preheader:                             ; preds = %if.else
  %wide.trip.count89 = zext nneg i32 %len to i64
  br label %for.body16

for.cond.cleanup15:                               ; preds = %for.body16
  %8 = tail call ptr asm sideeffect "larl $0, .LC202", "={r13}"() #4, !srcloc !9
  br label %for.cond.cleanup27

for.body16:                                       ; preds = %for.body16.preheader, %for.body16
  %indvars.iv86 = phi i64 [ 0, %for.body16.preheader ], [ %indvars.iv.next87, %for.body16 ]
  %arrayidx18 = getelementptr inbounds i32, ptr %0, i64 %indvars.iv86
  %9 = load i32, ptr %arrayidx18, align 4, !tbaa !5
  %call19 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %9)
  %indvars.iv.next87 = add nuw nsw i64 %indvars.iv86, 1
  %exitcond90.not = icmp eq i64 %indvars.iv.next87, %wide.trip.count89
  br i1 %exitcond90.not, label %for.cond.cleanup15, label %for.body16, !llvm.loop !10

for.cond.cleanup27:                               ; preds = %for.cond.cleanup15, %for.cond.cleanup15.thread
  tail call void @func3()
  unreachable

if.else36:                                        ; preds = %for.cond.cleanup
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.15)
  %10 = load i32, ptr %add.ptr, align 4, !tbaa !5
  %call38 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, i32 noundef signext %10)
  br i1 %cmp76, label %for.body44.preheader, label %for.cond.cleanup43

for.body44.preheader:                             ; preds = %if.else36
  %wide.trip.count = zext nneg i32 %len to i64
  br label %for.body44

for.cond.cleanup43:                               ; preds = %for.body44, %if.else36
  tail call void @func2()
  unreachable

for.body44:                                       ; preds = %for.body44.preheader, %for.body44
  %indvars.iv = phi i64 [ 0, %for.body44.preheader ], [ %indvars.iv.next, %for.body44 ]
  %arrayidx46 = getelementptr inbounds i32, ptr %0, i64 %indvars.iv
  %11 = load i32, ptr %arrayidx46, align 4, !tbaa !5
  %call47 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef signext %11)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup43, label %for.body44, !llvm.loop !12
}

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #4

; Function Attrs: nounwind
define dso_local signext range(i32 0, 2) i32 @main(i32 noundef signext %argc, ptr nocapture noundef readonly %argv) local_unnamed_addr #5 {
entry:
  %cmp = icmp slt i32 %argc, 2
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %arrayidx = getelementptr inbounds i8, ptr %argv, i64 8
  %0 = load ptr, ptr %arrayidx, align 8, !tbaa !13
  %call.i = tail call i64 @strtol(ptr nocapture noundef nonnull %0, ptr noundef null, i32 noundef signext 10) #4
  %1 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf1)
  %cmp3 = icmp eq i32 %1, 0
  br i1 %cmp3, label %if.then4, label %return

if.then4:                                         ; preds = %if.end
  %conv.i = trunc i64 %call.i to i32
  %cond = tail call i32 @llvm.smin.i32(i32 %conv.i, i32 10)
  %puts11 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.20)
  %call6 = tail call signext i32 @func1(i32 noundef signext %cond)
  unreachable

return:                                           ; preds = %if.end, %entry
  %str.19.sink = phi ptr [ @str.21, %entry ], [ @str.19, %if.end ]
  %retval.0 = phi i32 [ 1, %entry ], [ 0, %if.end ]
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) %str.19.sink)
  ret i32 %retval.0
}

; Function Attrs: mustprogress nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr nocapture noundef, i32 noundef signext) local_unnamed_addr #6

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #7

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #8

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #9

attributes #0 = { noinline noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { nofree nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #2 = { noreturn nounwind }
attributes #3 = { noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #4 = { nounwind }
attributes #5 = { nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #6 = { mustprogress nofree nounwind willreturn "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #7 = { nofree nounwind }
attributes #8 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git b289df99d26b008287e18cdb0858bc569de3f2ad)"}
!4 = !{i64 1669}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{i64 2347}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
!12 = distinct !{!12, !11}
!13 = !{!14, !14, i64 0}
!14 = !{!"any pointer", !7, i64 0}
