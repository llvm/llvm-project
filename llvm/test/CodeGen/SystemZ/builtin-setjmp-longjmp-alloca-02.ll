; Test for Frame Pointer in first slot in jmp_buf.
; Test assembly for nested setjmp for alloa.
; This test case takes input from stdin for size of alloca 
; and produce the right result.

; Frame Pointer in slot 1.
; Return address in slot 2.
; Stack Pointer in slot 4.

; RUN: llc < %s | FileCheck %s

; ModuleID = 'builtin-setjmp-longjmp-alloca-01.c'
source_filename = "builtin-setjmp-longjmp-alloca-01.c"
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

@buf3 = dso_local global [10 x ptr] zeroinitializer, align 8
@buf2 = dso_local global [10 x ptr] zeroinitializer, align 8
@buf1 = dso_local global [10 x ptr] zeroinitializer, align 8
@.str.3 = private unnamed_addr constant [22 x i8] c"Please enter length: \00", align 2
@.str.4 = private unnamed_addr constant [3 x i8] c"%d\00", align 2
@.str.8 = private unnamed_addr constant [9 x i8] c"arr: %d\0A\00", align 2
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
  %len = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %len) #5
  store i32 10, ptr %len, align 4, !tbaa !4
  %call = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3)
  %call1 = call signext i32 (ptr, ...) @__isoc99_scanf(ptr noundef nonnull @.str.4, ptr noundef nonnull %len)
  %0 = load i32, ptr %len, align 4, !tbaa !4
  %conv = sext i32 %0 to i64
  %mul = shl nsw i64 %conv, 2
  %1 = alloca i8, i64 %mul, align 8
  %cmp82 = icmp sgt i32 %0, 0
  br i1 %cmp82, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext nneg i32 %0 to i64
  %xtraiter = and i64 %wide.trip.count, 3
  %2 = icmp ult i32 %0, 4
  br i1 %2, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body.preheader.new

for.body.preheader.new:                           ; preds = %for.body.preheader
  %unroll_iter = and i64 %wide.trip.count, 2147483644
  br label %for.body

for.cond.cleanup.loopexit.unr-lcssa:              ; preds = %for.body, %for.body.preheader
  %indvars.iv.unr = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next.3, %for.body ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %for.cond.cleanup, label %for.body.epil

for.body.epil:                                    ; preds = %for.cond.cleanup.loopexit.unr-lcssa, %for.body.epil
  %indvars.iv.epil = phi i64 [ %indvars.iv.next.epil, %for.body.epil ], [ %indvars.iv.unr, %for.cond.cleanup.loopexit.unr-lcssa ]
  %epil.iter = phi i64 [ %epil.iter.next, %for.body.epil ], [ 0, %for.cond.cleanup.loopexit.unr-lcssa ]
  %indvars.iv.next.epil = add nuw nsw i64 %indvars.iv.epil, 1
  %indvars.epil = trunc i64 %indvars.iv.next.epil to i32
  %3 = trunc nuw nsw i64 %indvars.iv.epil to i32
  %add.epil = mul i32 %indvars.epil, %3
  %arrayidx.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.epil
  store i32 %add.epil, ptr %arrayidx.epil, align 4, !tbaa !4
  %epil.iter.next = add i64 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %for.cond.cleanup, label %for.body.epil, !llvm.loop !8

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit.unr-lcssa, %for.body.epil, %entry
; CHECK:        larl    %r0, .LBB3_13
; CHECK:        stg     %r0, 8(%r1)
; CHECK:        stg     %r11, 0(%r1)
; CHECK:        stg     %r15, 24(%r1)

  %4 = call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf2)
  %cmp4 = icmp eq i32 %4, 0
  br i1 %cmp4, label %if.then, label %if.else40

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %indvars.iv = phi i64 [ 0, %for.body.preheader.new ], [ %indvars.iv.next.3, %for.body ]
  %niter = phi i64 [ 0, %for.body.preheader.new ], [ %niter.next.3, %for.body ]
  %indvars.iv.next = or disjoint i64 %indvars.iv, 1
  %indvars = trunc i64 %indvars.iv.next to i32
  %5 = trunc nuw nsw i64 %indvars.iv to i32
  %add = mul i32 %indvars, %5
  %arrayidx = getelementptr inbounds i32, ptr %1, i64 %indvars.iv
  store i32 %add, ptr %arrayidx, align 8, !tbaa !4
  %indvars.iv.next.1 = or disjoint i64 %indvars.iv, 2
  %indvars.1 = trunc i64 %indvars.iv.next.1 to i32
  %6 = trunc nuw nsw i64 %indvars.iv.next to i32
  %add.1 = mul i32 %indvars.1, %6
  %arrayidx.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next
  store i32 %add.1, ptr %arrayidx.1, align 4, !tbaa !4
  %indvars.iv.next.2 = or disjoint i64 %indvars.iv, 3
  %indvars.2 = trunc i64 %indvars.iv.next.2 to i32
  %7 = trunc nuw nsw i64 %indvars.iv.next.1 to i32
  %add.2 = mul i32 %indvars.2, %7
  %arrayidx.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next.1
  store i32 %add.2, ptr %arrayidx.2, align 8, !tbaa !4
  %indvars.iv.next.3 = add nuw nsw i64 %indvars.iv, 4
  %indvars.3 = trunc i64 %indvars.iv.next.3 to i32
  %8 = trunc nuw nsw i64 %indvars.iv.next.2 to i32
  %add.3 = mul i32 %indvars.3, %8
  %arrayidx.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next.2
  store i32 %add.3, ptr %arrayidx.3, align 4, !tbaa !4
  %niter.next.3 = add i64 %niter, 4
  %niter.ncmp.3 = icmp eq i64 %niter.next.3, %unroll_iter
  br i1 %niter.ncmp.3, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body, !llvm.loop !10

if.then:                                          ; preds = %for.cond.cleanup
  %puts75 = call i32 @puts(ptr nonnull dereferenceable(1) @str.15)
  %9 = call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf3)
  %cmp7 = icmp eq i32 %9, 0
  br i1 %cmp7, label %if.then9, label %if.else

if.then9:                                         ; preds = %if.then
  %puts80 = call i32 @puts(ptr nonnull dereferenceable(1) @str.17)
  call void @func4()
  unreachable

if.else:                                          ; preds = %if.then
  %puts76 = call i32 @puts(ptr nonnull dereferenceable(1) @str.16)
  %10 = load i32, ptr %len, align 4, !tbaa !4
  %cmp1486 = icmp sgt i32 %10, 0
  br i1 %cmp1486, label %for.body17, label %for.cond.cleanup28

for.cond25.preheader:                             ; preds = %for.body17
  %cmp2688 = icmp sgt i32 %13, 0
  br i1 %cmp2688, label %for.body29.preheader, label %for.cond.cleanup28

for.body29.preheader:                             ; preds = %for.cond25.preheader
  %wide.trip.count104 = zext nneg i32 %13 to i64
  %xtraiter108 = and i64 %wide.trip.count104, 3
  %11 = icmp ult i32 %13, 4
  br i1 %11, label %for.cond.cleanup28.loopexit.unr-lcssa, label %for.body29.preheader.new

for.body29.preheader.new:                         ; preds = %for.body29.preheader
  %unroll_iter111 = and i64 %wide.trip.count104, 2147483644
  br label %for.body29

for.body17:                                       ; preds = %if.else, %for.body17
  %indvars.iv96 = phi i64 [ %indvars.iv.next97, %for.body17 ], [ 0, %if.else ]
  %arrayidx19 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv96
  %12 = load i32, ptr %arrayidx19, align 4, !tbaa !4
  %call20 = call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef signext %12)
  %indvars.iv.next97 = add nuw nsw i64 %indvars.iv96, 1
  %13 = load i32, ptr %len, align 4, !tbaa !4
  %14 = sext i32 %13 to i64
  %cmp14 = icmp slt i64 %indvars.iv.next97, %14
  br i1 %cmp14, label %for.body17, label %for.cond25.preheader, !llvm.loop !12

for.cond.cleanup28.loopexit.unr-lcssa:            ; preds = %for.body29, %for.body29.preheader
  %indvars.iv100.unr = phi i64 [ 0, %for.body29.preheader ], [ %indvars.iv.next101.3, %for.body29 ]
  %lcmp.mod110.not = icmp eq i64 %xtraiter108, 0
  br i1 %lcmp.mod110.not, label %for.cond.cleanup28, label %for.body29.epil

for.body29.epil:                                  ; preds = %for.cond.cleanup28.loopexit.unr-lcssa, %for.body29.epil
  %indvars.iv100.epil = phi i64 [ %indvars.iv.next101.epil, %for.body29.epil ], [ %indvars.iv100.unr, %for.cond.cleanup28.loopexit.unr-lcssa ]
  %epil.iter109 = phi i64 [ %epil.iter109.next, %for.body29.epil ], [ 0, %for.cond.cleanup28.loopexit.unr-lcssa ]
  %indvars.iv.next101.epil = add nuw nsw i64 %indvars.iv100.epil, 1
  %indvars102.epil = trunc i64 %indvars.iv.next101.epil to i32
  %15 = trunc nuw nsw i64 %indvars.iv100.epil to i32
  %mul3177.epil = mul i32 %indvars102.epil, %15
  %add3379.epil = add nuw nsw i32 %mul3177.epil, 1
  %add34.epil = mul i32 %add3379.epil, %15
  %arrayidx36.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv100.epil
  store i32 %add34.epil, ptr %arrayidx36.epil, align 4, !tbaa !4
  %epil.iter109.next = add i64 %epil.iter109, 1
  %epil.iter109.cmp.not = icmp eq i64 %epil.iter109.next, %xtraiter108
  br i1 %epil.iter109.cmp.not, label %for.cond.cleanup28, label %for.body29.epil, !llvm.loop !13

for.cond.cleanup28:                               ; preds = %for.cond.cleanup28.loopexit.unr-lcssa, %for.body29.epil, %if.else, %for.cond25.preheader
  call void @func3()
  unreachable

for.body29:                                       ; preds = %for.body29, %for.body29.preheader.new
  %indvars.iv100 = phi i64 [ 0, %for.body29.preheader.new ], [ %indvars.iv.next101.3, %for.body29 ]
  %niter112 = phi i64 [ 0, %for.body29.preheader.new ], [ %niter112.next.3, %for.body29 ]
  %indvars.iv.next101 = or disjoint i64 %indvars.iv100, 1
  %indvars102 = trunc i64 %indvars.iv.next101 to i32
  %16 = trunc nuw nsw i64 %indvars.iv100 to i32
  %mul3177 = mul i32 %indvars102, %16
  %add3379 = or disjoint i32 %mul3177, 1
  %add34 = mul i32 %add3379, %16
  %arrayidx36 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv100
  store i32 %add34, ptr %arrayidx36, align 8, !tbaa !4
  %indvars.iv.next101.1 = or disjoint i64 %indvars.iv100, 2
  %indvars102.1 = trunc i64 %indvars.iv.next101.1 to i32
  %17 = trunc nuw nsw i64 %indvars.iv.next101 to i32
  %mul3177.1 = mul i32 %indvars102.1, %17
  %add3379.1 = or disjoint i32 %mul3177.1, 1
  %add34.1 = mul i32 %add3379.1, %17
  %arrayidx36.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next101
  store i32 %add34.1, ptr %arrayidx36.1, align 4, !tbaa !4
  %indvars.iv.next101.2 = or disjoint i64 %indvars.iv100, 3
  %indvars102.2 = trunc i64 %indvars.iv.next101.2 to i32
  %18 = trunc nuw nsw i64 %indvars.iv.next101.1 to i32
  %mul3177.2 = mul i32 %indvars102.2, %18
  %add3379.2 = or disjoint i32 %mul3177.2, 1
  %add34.2 = mul i32 %add3379.2, %18
  %arrayidx36.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next101.1
  store i32 %add34.2, ptr %arrayidx36.2, align 8, !tbaa !4
  %indvars.iv.next101.3 = add nuw nsw i64 %indvars.iv100, 4
  %indvars102.3 = trunc i64 %indvars.iv.next101.3 to i32
  %19 = trunc nuw nsw i64 %indvars.iv.next101.2 to i32
  %mul3177.3 = mul i32 %indvars102.3, %19
  %add3379.3 = or disjoint i32 %mul3177.3, 1
  %add34.3 = mul i32 %add3379.3, %19
  %arrayidx36.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next101.2
  store i32 %add34.3, ptr %arrayidx36.3, align 4, !tbaa !4
  %niter112.next.3 = add i64 %niter112, 4
  %niter112.ncmp.3 = icmp eq i64 %niter112.next.3, %unroll_iter111
  br i1 %niter112.ncmp.3, label %for.cond.cleanup28.loopexit.unr-lcssa, label %for.body29, !llvm.loop !14

if.else40:                                        ; preds = %for.cond.cleanup
  %puts = call i32 @puts(ptr nonnull dereferenceable(1) @str.14)
  %20 = load i32, ptr %len, align 4, !tbaa !4
  %cmp4484 = icmp sgt i32 %20, 0
  br i1 %cmp4484, label %for.body47, label %for.cond.cleanup46

for.cond.cleanup46:                               ; preds = %for.body47, %if.else40
  call void @func2()
  unreachable

for.body47:                                       ; preds = %if.else40, %for.body47
  %indvars.iv92 = phi i64 [ %indvars.iv.next93, %for.body47 ], [ 0, %if.else40 ]
  %arrayidx49 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv92
  %21 = load i32, ptr %arrayidx49, align 4, !tbaa !4
  %call50 = call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef signext %21)
  %indvars.iv.next93 = add nuw nsw i64 %indvars.iv92, 1
  %22 = load i32, ptr %len, align 4, !tbaa !4
  %23 = sext i32 %22 to i64
  %cmp44 = icmp slt i64 %indvars.iv.next93, %23
  br i1 %cmp44, label %for.body47, label %for.cond.cleanup46, !llvm.loop !15
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #4

; Function Attrs: nofree nounwind
declare noundef signext i32 @__isoc99_scanf(ptr nocapture noundef readonly, ...) local_unnamed_addr #1

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #5

; Function Attrs: nounwind
define dso_local noundef signext i32 @main() local_unnamed_addr #6 {
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
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #7

attributes #0 = { noinline noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #2 = { noreturn nounwind }
attributes #3 = { noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nounwind }
attributes #6 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #7 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git 79880371396d6e486bf6bacd6c4087ebdac591f8)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.unroll.disable"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
!12 = distinct !{!12, !11}
!13 = distinct !{!13, !9}
!14 = distinct !{!14, !11}
!15 = distinct !{!15, !11}
