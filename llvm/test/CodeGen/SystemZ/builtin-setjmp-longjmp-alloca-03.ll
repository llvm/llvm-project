; This test case produces right result when alloca  len is not local variable.
; Test output of setjmp/longjmp with nested setjmp for alloca
; Test for Frame Pointer in first slot in jmp_buf.

; RUN: clang -O2 -o %t %s
; RUN: %t | FileCheck %s

; ModuleID = 'builtin-setjmp-longjmp-alloca-03.c'
source_filename = "builtin-setjmp-longjmp-alloca-03.c"
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

@buf3 = dso_local global [10 x ptr] zeroinitializer, align 8
@buf2 = dso_local global [10 x ptr] zeroinitializer, align 8
@buf1 = dso_local global [10 x ptr] zeroinitializer, align 8
@len = dso_local local_unnamed_addr global i32 10, align 4
@.str.11 = private unnamed_addr constant [9 x i8] c"arr: %d\0A\00", align 2
@str = private unnamed_addr constant [9 x i8] c"In func4\00", align 1
@str.15 = private unnamed_addr constant [9 x i8] c"In func3\00", align 1
@str.16 = private unnamed_addr constant [9 x i8] c"In func2\00", align 1
@str.17 = private unnamed_addr constant [20 x i8] c"Returned from func3\00", align 1
@str.18 = private unnamed_addr constant [32 x i8] c"First __builtin_setjmp in func1\00", align 1
@str.19 = private unnamed_addr constant [20 x i8] c"Returned from func4\00", align 1
@str.25 = private unnamed_addr constant [33 x i8] c"Second __builtin_setjmp in func1\00", align 1
@str.26 = private unnamed_addr constant [8 x i8] c"case 4:\00", align 1
@str.27 = private unnamed_addr constant [8 x i8] c"case 3:\00", align 1
@str.28 = private unnamed_addr constant [8 x i8] c"case 2:\00", align 1
@str.29 = private unnamed_addr constant [8 x i8] c"case 1:\00", align 1
@str.30 = private unnamed_addr constant [8 x i8] c"case 0:\00", align 1
@str.31 = private unnamed_addr constant [44 x i8] c"In main, after __builtin_longjmp from func1\00", align 1
@str.32 = private unnamed_addr constant [20 x i8] c"In main, first time\00", align 1

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
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.15)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf2)
  unreachable
}

; Function Attrs: noinline noreturn nounwind
define dso_local void @func2() local_unnamed_addr #0 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.16)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf1)
  unreachable
}

; Function Attrs: noreturn nounwind
define dso_local noundef signext i32 @func1() local_unnamed_addr #3 {
entry:
  %0 = load i32, ptr @len, align 4, !tbaa !4
  %conv = sext i32 %0 to i64
  %mul = shl nsw i64 %conv, 2
  %1 = alloca i8, i64 %mul, align 8
  %cmp348 = icmp sgt i32 %0, 0
  br i1 %cmp348, label %for.body.preheader, label %for.cond.cleanup

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
; CHECK: First __builtin_setjmp in func1
; CHECK: Second __builtin_setjmp in func1
; CHECK: case 0:
; CHECK: Returned from func4
; CHECK: arr: 0
; CHECK: arr: 4
; CHECK: arr: 10
; CHECK: arr: 18
; CHECK: arr: 28
; CHECK: arr: 40
; CHECK: arr: 54
; CHECK: arr: 70
; CHECK: arr: 88
; CHECK: arr: 108
; CHECK: case 0:
; CHECK: Returned from func3
; CHECK: arr: 0
; CHECK: arr: 5
; CHECK: arr: 26
; CHECK: arr: 99
; CHECK: arr: 284
; CHECK: arr: 665
; CHECK: arr: 1350
; CHECK: arr: 2471
; CHECK: arr: 4184
; CHECK: arr: 6669

  %4 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf2)
  %cmp3 = icmp eq i32 %4, 0
  br i1 %cmp3, label %if.then, label %if.else200

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
  %puts305 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.18)
  %9 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf3)
  %cmp5 = icmp eq i32 %9, 0
  br i1 %cmp5, label %if.then7, label %if.else

if.then7:                                         ; preds = %if.then
  %puts322 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.25)
  %10 = load i32, ptr @len, align 4, !tbaa !4
  %rem = srem i32 %10, 5
  switch i32 %rem, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb24
    i32 2, label %sw.bb43
    i32 3, label %sw.bb61
    i32 4, label %sw.bb77
  ]

sw.bb:                                            ; preds = %if.then7
  %puts337 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %11 = load i32, ptr @len, align 4, !tbaa !4
  %cmp12372 = icmp sgt i32 %11, 0
  br i1 %cmp12372, label %for.body15.preheader, label %sw.epilog

for.body15.preheader:                             ; preds = %sw.bb
  %wide.trip.count458 = zext nneg i32 %11 to i64
  %xtraiter532 = and i64 %wide.trip.count458, 3
  %12 = icmp ult i32 %11, 4
  br i1 %12, label %sw.epilog.loopexit.unr-lcssa, label %for.body15.preheader.new

for.body15.preheader.new:                         ; preds = %for.body15.preheader
  %unroll_iter535 = and i64 %wide.trip.count458, 2147483644
  br label %for.body15

for.body15:                                       ; preds = %for.body15, %for.body15.preheader.new
  %indvars.iv453 = phi i64 [ 0, %for.body15.preheader.new ], [ %indvars.iv.next454.3, %for.body15 ]
  %niter536 = phi i64 [ 0, %for.body15.preheader.new ], [ %niter536.next.3, %for.body15 ]
  %indvars457 = trunc i64 %indvars.iv453 to i32
  %mul17338 = or disjoint i32 %indvars457, 3
  %add18 = mul i32 %mul17338, %indvars457
  %arrayidx20 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv453
  store i32 %add18, ptr %arrayidx20, align 8, !tbaa !4
  %indvars.iv.next454 = or disjoint i64 %indvars.iv453, 1
  %indvars457.1 = trunc i64 %indvars.iv.next454 to i32
  %mul17338.1 = add nuw i32 %indvars457.1, 3
  %add18.1 = mul i32 %mul17338.1, %indvars457.1
  %arrayidx20.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next454
  store i32 %add18.1, ptr %arrayidx20.1, align 4, !tbaa !4
  %indvars.iv.next454.1 = or disjoint i64 %indvars.iv453, 2
  %indvars457.2 = trunc i64 %indvars.iv.next454.1 to i32
  %mul17338.2 = add nuw i32 %indvars457.2, 3
  %add18.2 = mul i32 %mul17338.2, %indvars457.2
  %arrayidx20.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next454.1
  store i32 %add18.2, ptr %arrayidx20.2, align 8, !tbaa !4
  %indvars.iv.next454.2 = or disjoint i64 %indvars.iv453, 3
  %indvars457.3 = trunc i64 %indvars.iv.next454.2 to i32
  %mul17338.3 = add nuw i32 %indvars457.3, 3
  %add18.3 = mul i32 %mul17338.3, %indvars457.3
  %arrayidx20.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next454.2
  store i32 %add18.3, ptr %arrayidx20.3, align 4, !tbaa !4
  %indvars.iv.next454.3 = add nuw nsw i64 %indvars.iv453, 4
  %niter536.next.3 = add i64 %niter536, 4
  %niter536.ncmp.3 = icmp eq i64 %niter536.next.3, %unroll_iter535
  br i1 %niter536.ncmp.3, label %sw.epilog.loopexit.unr-lcssa, label %for.body15, !llvm.loop !12

sw.bb24:                                          ; preds = %if.then7
  %puts333 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.29)
  %13 = load i32, ptr @len, align 4, !tbaa !4
  %cmp28370 = icmp sgt i32 %13, 0
  br i1 %cmp28370, label %for.body31.preheader, label %sw.epilog

for.body31.preheader:                             ; preds = %sw.bb24
  %wide.trip.count451 = zext nneg i32 %13 to i64
  %xtraiter527 = and i64 %wide.trip.count451, 3
  %14 = icmp ult i32 %13, 4
  br i1 %14, label %sw.epilog.loopexit478.unr-lcssa, label %for.body31.preheader.new

for.body31.preheader.new:                         ; preds = %for.body31.preheader
  %unroll_iter530 = and i64 %wide.trip.count451, 2147483644
  br label %for.body31

for.body31:                                       ; preds = %for.body31, %for.body31.preheader.new
  %indvars.iv447 = phi i64 [ 0, %for.body31.preheader.new ], [ %indvars.iv.next448.3, %for.body31 ]
  %niter531 = phi i64 [ 0, %for.body31.preheader.new ], [ %niter531.next.3, %for.body31 ]
  %indvars.iv.next448 = or disjoint i64 %indvars.iv447, 1
  %indvars449 = trunc i64 %indvars.iv.next448 to i32
  %15 = trunc nuw nsw i64 %indvars.iv447 to i32
  %mul33334 = mul i32 %indvars449, %15
  %add35336 = or disjoint i32 %mul33334, 3
  %add37 = mul i32 %add35336, %15
  %arrayidx39 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv447
  store i32 %add37, ptr %arrayidx39, align 8, !tbaa !4
  %indvars.iv.next448.1 = or disjoint i64 %indvars.iv447, 2
  %indvars449.1 = trunc i64 %indvars.iv.next448.1 to i32
  %16 = trunc nuw nsw i64 %indvars.iv.next448 to i32
  %mul33334.1 = mul i32 %indvars449.1, %16
  %add35336.1 = add i32 %mul33334.1, 3
  %add37.1 = mul i32 %add35336.1, %16
  %arrayidx39.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next448
  store i32 %add37.1, ptr %arrayidx39.1, align 4, !tbaa !4
  %indvars.iv.next448.2 = or disjoint i64 %indvars.iv447, 3
  %indvars449.2 = trunc i64 %indvars.iv.next448.2 to i32
  %17 = trunc nuw nsw i64 %indvars.iv.next448.1 to i32
  %mul33334.2 = mul i32 %indvars449.2, %17
  %add35336.2 = add i32 %mul33334.2, 3
  %add37.2 = mul i32 %add35336.2, %17
  %arrayidx39.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next448.1
  store i32 %add37.2, ptr %arrayidx39.2, align 8, !tbaa !4
  %indvars.iv.next448.3 = add nuw nsw i64 %indvars.iv447, 4
  %indvars449.3 = trunc i64 %indvars.iv.next448.3 to i32
  %18 = trunc nuw nsw i64 %indvars.iv.next448.2 to i32
  %mul33334.3 = mul i32 %indvars449.3, %18
  %add35336.3 = or disjoint i32 %mul33334.3, 3
  %add37.3 = mul i32 %add35336.3, %18
  %arrayidx39.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next448.2
  store i32 %add37.3, ptr %arrayidx39.3, align 4, !tbaa !4
  %niter531.next.3 = add i64 %niter531, 4
  %niter531.ncmp.3 = icmp eq i64 %niter531.next.3, %unroll_iter530
  br i1 %niter531.ncmp.3, label %sw.epilog.loopexit478.unr-lcssa, label %for.body31, !llvm.loop !13

sw.bb43:                                          ; preds = %if.then7
  %puts329 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.28)
  %19 = load i32, ptr @len, align 4, !tbaa !4
  %cmp47368 = icmp sgt i32 %19, 0
  br i1 %cmp47368, label %for.body50.preheader, label %sw.epilog

for.body50.preheader:                             ; preds = %sw.bb43
  %wide.trip.count445 = zext nneg i32 %19 to i64
  %xtraiter522 = and i64 %wide.trip.count445, 3
  %20 = icmp ult i32 %19, 4
  br i1 %20, label %sw.epilog.loopexit479.unr-lcssa, label %for.body50.preheader.new

for.body50.preheader.new:                         ; preds = %for.body50.preheader
  %unroll_iter525 = and i64 %wide.trip.count445, 2147483644
  br label %for.body50

for.body50:                                       ; preds = %for.body50, %for.body50.preheader.new
  %indvars.iv440 = phi i64 [ 0, %for.body50.preheader.new ], [ %indvars.iv.next441.3, %for.body50 ]
  %niter526 = phi i64 [ 0, %for.body50.preheader.new ], [ %niter526.next.3, %for.body50 ]
  %indvars444 = trunc i64 %indvars.iv440 to i32
  %i45.0331 = add nsw i32 %indvars444, -1
  %mul52330 = mul i32 %i45.0331, %indvars444
  %sub332 = or disjoint i32 %mul52330, 3
  %add55 = mul i32 %sub332, %indvars444
  %arrayidx57 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv440
  store i32 %add55, ptr %arrayidx57, align 8, !tbaa !4
  %indvars.iv.next441 = or disjoint i64 %indvars.iv440, 1
  %indvars444.1 = trunc i64 %indvars.iv.next441 to i32
  %i45.0331.1 = add nsw i32 %indvars444.1, -1
  %mul52330.1 = mul i32 %i45.0331.1, %indvars444.1
  %sub332.1 = or disjoint i32 %mul52330.1, 3
  %add55.1 = mul i32 %sub332.1, %indvars444.1
  %arrayidx57.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next441
  store i32 %add55.1, ptr %arrayidx57.1, align 4, !tbaa !4
  %indvars.iv.next441.1 = or disjoint i64 %indvars.iv440, 2
  %indvars444.2 = trunc i64 %indvars.iv.next441.1 to i32
  %i45.0331.2 = add nsw i32 %indvars444.2, -1
  %mul52330.2 = mul i32 %i45.0331.2, %indvars444.2
  %sub332.2 = add i32 %mul52330.2, 3
  %add55.2 = mul i32 %sub332.2, %indvars444.2
  %arrayidx57.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next441.1
  store i32 %add55.2, ptr %arrayidx57.2, align 8, !tbaa !4
  %indvars.iv.next441.2 = or disjoint i64 %indvars.iv440, 3
  %indvars444.3 = trunc i64 %indvars.iv.next441.2 to i32
  %i45.0331.3 = add nsw i32 %indvars444.3, -1
  %mul52330.3 = mul i32 %i45.0331.3, %indvars444.3
  %sub332.3 = add i32 %mul52330.3, 3
  %add55.3 = mul i32 %sub332.3, %indvars444.3
  %arrayidx57.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next441.2
  store i32 %add55.3, ptr %arrayidx57.3, align 4, !tbaa !4
  %indvars.iv.next441.3 = add nuw nsw i64 %indvars.iv440, 4
  %niter526.next.3 = add i64 %niter526, 4
  %niter526.ncmp.3 = icmp eq i64 %niter526.next.3, %unroll_iter525
  br i1 %niter526.ncmp.3, label %sw.epilog.loopexit479.unr-lcssa, label %for.body50, !llvm.loop !14

sw.bb61:                                          ; preds = %if.then7
  %puts327 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.27)
  %21 = load i32, ptr @len, align 4, !tbaa !4
  %cmp65366 = icmp sgt i32 %21, 0
  br i1 %cmp65366, label %for.body68.preheader, label %sw.epilog

for.body68.preheader:                             ; preds = %sw.bb61
  %wide.trip.count438 = zext nneg i32 %21 to i64
  %xtraiter517 = and i64 %wide.trip.count438, 3
  %22 = icmp ult i32 %21, 4
  br i1 %22, label %sw.epilog.loopexit480.unr-lcssa, label %for.body68.preheader.new

for.body68.preheader.new:                         ; preds = %for.body68.preheader
  %unroll_iter520 = and i64 %wide.trip.count438, 2147483644
  br label %for.body68

for.body68:                                       ; preds = %for.body68, %for.body68.preheader.new
  %indvars.iv434 = phi i64 [ 0, %for.body68.preheader.new ], [ %indvars.iv.next435.3, %for.body68 ]
  %niter521 = phi i64 [ 0, %for.body68.preheader.new ], [ %niter521.next.3, %for.body68 ]
  %indvars.iv.next435 = or disjoint i64 %indvars.iv434, 1
  %indvars436 = trunc i64 %indvars.iv.next435 to i32
  %23 = trunc nuw nsw i64 %indvars.iv434 to i32
  %add70 = mul i32 %indvars436, %23
  %add71 = or disjoint i32 %add70, 3
  %arrayidx73 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv434
  store i32 %add71, ptr %arrayidx73, align 8, !tbaa !4
  %indvars.iv.next435.1 = or disjoint i64 %indvars.iv434, 2
  %indvars436.1 = trunc i64 %indvars.iv.next435.1 to i32
  %24 = trunc nuw nsw i64 %indvars.iv.next435 to i32
  %add70.1 = mul i32 %indvars436.1, %24
  %add71.1 = add nsw i32 %add70.1, 3
  %arrayidx73.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next435
  store i32 %add71.1, ptr %arrayidx73.1, align 4, !tbaa !4
  %indvars.iv.next435.2 = or disjoint i64 %indvars.iv434, 3
  %indvars436.2 = trunc i64 %indvars.iv.next435.2 to i32
  %25 = trunc nuw nsw i64 %indvars.iv.next435.1 to i32
  %add70.2 = mul i32 %indvars436.2, %25
  %add71.2 = add nsw i32 %add70.2, 3
  %arrayidx73.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next435.1
  store i32 %add71.2, ptr %arrayidx73.2, align 8, !tbaa !4
  %indvars.iv.next435.3 = add nuw nsw i64 %indvars.iv434, 4
  %indvars436.3 = trunc i64 %indvars.iv.next435.3 to i32
  %26 = trunc nuw nsw i64 %indvars.iv.next435.2 to i32
  %add70.3 = mul i32 %indvars436.3, %26
  %add71.3 = or disjoint i32 %add70.3, 3
  %arrayidx73.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next435.2
  store i32 %add71.3, ptr %arrayidx73.3, align 4, !tbaa !4
  %niter521.next.3 = add i64 %niter521, 4
  %niter521.ncmp.3 = icmp eq i64 %niter521.next.3, %unroll_iter520
  br i1 %niter521.ncmp.3, label %sw.epilog.loopexit480.unr-lcssa, label %for.body68, !llvm.loop !15

sw.bb77:                                          ; preds = %if.then7
  %puts323 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.26)
  %27 = load i32, ptr @len, align 4, !tbaa !4
  %cmp81364 = icmp sgt i32 %27, 0
  br i1 %cmp81364, label %for.body84.preheader, label %sw.epilog

for.body84.preheader:                             ; preds = %sw.bb77
  %wide.trip.count432 = zext nneg i32 %27 to i64
  %xtraiter512 = and i64 %wide.trip.count432, 3
  %28 = icmp ult i32 %27, 4
  br i1 %28, label %sw.epilog.loopexit481.unr-lcssa, label %for.body84.preheader.new

for.body84.preheader.new:                         ; preds = %for.body84.preheader
  %unroll_iter515 = and i64 %wide.trip.count432, 2147483644
  br label %for.body84

for.body84:                                       ; preds = %for.body84, %for.body84.preheader.new
  %indvars.iv426 = phi i64 [ 0, %for.body84.preheader.new ], [ %indvars.iv.next427.3, %for.body84 ]
  %niter516 = phi i64 [ 0, %for.body84.preheader.new ], [ %niter516.next.3, %for.body84 ]
  %indvars431 = trunc i64 %indvars.iv426 to i32
  %mul85 = mul nuw nsw i32 %indvars431, %indvars431
  %mul86325 = or disjoint i32 %mul85, 1
  %mul87324 = mul i32 %mul86325, %indvars431
  %add89326 = or disjoint i32 %mul87324, 3
  %add91 = mul i32 %add89326, %indvars431
  %arrayidx93 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv426
  store i32 %add91, ptr %arrayidx93, align 8, !tbaa !4
  %indvars.iv.next427 = or disjoint i64 %indvars.iv426, 1
  %indvars431.1 = trunc i64 %indvars.iv.next427 to i32
  %mul85.1 = mul nuw nsw i32 %indvars431.1, %indvars431.1
  %mul86325.1 = add nuw nsw i32 %mul85.1, 1
  %mul87324.1 = mul i32 %mul86325.1, %indvars431.1
  %add89326.1 = add i32 %mul87324.1, 3
  %add91.1 = mul i32 %add89326.1, %indvars431.1
  %arrayidx93.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next427
  store i32 %add91.1, ptr %arrayidx93.1, align 4, !tbaa !4
  %indvars.iv.next427.1 = or disjoint i64 %indvars.iv426, 2
  %indvars431.2 = trunc i64 %indvars.iv.next427.1 to i32
  %mul85.2 = mul nuw nsw i32 %indvars431.2, %indvars431.2
  %mul86325.2 = or disjoint i32 %mul85.2, 1
  %mul87324.2 = mul i32 %mul86325.2, %indvars431.2
  %add89326.2 = add i32 %mul87324.2, 3
  %add91.2 = mul i32 %add89326.2, %indvars431.2
  %arrayidx93.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next427.1
  store i32 %add91.2, ptr %arrayidx93.2, align 8, !tbaa !4
  %indvars.iv.next427.2 = or disjoint i64 %indvars.iv426, 3
  %indvars431.3 = trunc i64 %indvars.iv.next427.2 to i32
  %mul85.3 = mul nuw nsw i32 %indvars431.3, %indvars431.3
  %mul86325.3 = add nuw nsw i32 %mul85.3, 1
  %mul87324.3 = mul i32 %mul86325.3, %indvars431.3
  %add89326.3 = add i32 %mul87324.3, 3
  %add91.3 = mul i32 %add89326.3, %indvars431.3
  %arrayidx93.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next427.2
  store i32 %add91.3, ptr %arrayidx93.3, align 4, !tbaa !4
  %indvars.iv.next427.3 = add nuw nsw i64 %indvars.iv426, 4
  %niter516.next.3 = add i64 %niter516, 4
  %niter516.ncmp.3 = icmp eq i64 %niter516.next.3, %unroll_iter515
  br i1 %niter516.ncmp.3, label %sw.epilog.loopexit481.unr-lcssa, label %for.body84, !llvm.loop !16

sw.epilog.loopexit.unr-lcssa:                     ; preds = %for.body15, %for.body15.preheader
  %indvars.iv453.unr = phi i64 [ 0, %for.body15.preheader ], [ %indvars.iv.next454.3, %for.body15 ]
  %lcmp.mod534.not = icmp eq i64 %xtraiter532, 0
  br i1 %lcmp.mod534.not, label %sw.epilog, label %for.body15.epil

for.body15.epil:                                  ; preds = %sw.epilog.loopexit.unr-lcssa, %for.body15.epil
  %indvars.iv453.epil = phi i64 [ %indvars.iv.next454.epil, %for.body15.epil ], [ %indvars.iv453.unr, %sw.epilog.loopexit.unr-lcssa ]
  %epil.iter533 = phi i64 [ %epil.iter533.next, %for.body15.epil ], [ 0, %sw.epilog.loopexit.unr-lcssa ]
  %indvars457.epil = trunc i64 %indvars.iv453.epil to i32
  %mul17338.epil = add nuw i32 %indvars457.epil, 3
  %add18.epil = mul i32 %mul17338.epil, %indvars457.epil
  %arrayidx20.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv453.epil
  store i32 %add18.epil, ptr %arrayidx20.epil, align 4, !tbaa !4
  %indvars.iv.next454.epil = add nuw nsw i64 %indvars.iv453.epil, 1
  %epil.iter533.next = add i64 %epil.iter533, 1
  %epil.iter533.cmp.not = icmp eq i64 %epil.iter533.next, %xtraiter532
  br i1 %epil.iter533.cmp.not, label %sw.epilog, label %for.body15.epil, !llvm.loop !17

sw.epilog.loopexit478.unr-lcssa:                  ; preds = %for.body31, %for.body31.preheader
  %indvars.iv447.unr = phi i64 [ 0, %for.body31.preheader ], [ %indvars.iv.next448.3, %for.body31 ]
  %lcmp.mod529.not = icmp eq i64 %xtraiter527, 0
  br i1 %lcmp.mod529.not, label %sw.epilog, label %for.body31.epil

for.body31.epil:                                  ; preds = %sw.epilog.loopexit478.unr-lcssa, %for.body31.epil
  %indvars.iv447.epil = phi i64 [ %indvars.iv.next448.epil, %for.body31.epil ], [ %indvars.iv447.unr, %sw.epilog.loopexit478.unr-lcssa ]
  %epil.iter528 = phi i64 [ %epil.iter528.next, %for.body31.epil ], [ 0, %sw.epilog.loopexit478.unr-lcssa ]
  %indvars.iv.next448.epil = add nuw nsw i64 %indvars.iv447.epil, 1
  %indvars449.epil = trunc i64 %indvars.iv.next448.epil to i32
  %29 = trunc nuw nsw i64 %indvars.iv447.epil to i32
  %mul33334.epil = mul i32 %indvars449.epil, %29
  %add35336.epil = add i32 %mul33334.epil, 3
  %add37.epil = mul i32 %add35336.epil, %29
  %arrayidx39.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv447.epil
  store i32 %add37.epil, ptr %arrayidx39.epil, align 4, !tbaa !4
  %epil.iter528.next = add i64 %epil.iter528, 1
  %epil.iter528.cmp.not = icmp eq i64 %epil.iter528.next, %xtraiter527
  br i1 %epil.iter528.cmp.not, label %sw.epilog, label %for.body31.epil, !llvm.loop !18

sw.epilog.loopexit479.unr-lcssa:                  ; preds = %for.body50, %for.body50.preheader
  %indvars.iv440.unr = phi i64 [ 0, %for.body50.preheader ], [ %indvars.iv.next441.3, %for.body50 ]
  %lcmp.mod524.not = icmp eq i64 %xtraiter522, 0
  br i1 %lcmp.mod524.not, label %sw.epilog, label %for.body50.epil

for.body50.epil:                                  ; preds = %sw.epilog.loopexit479.unr-lcssa, %for.body50.epil
  %indvars.iv440.epil = phi i64 [ %indvars.iv.next441.epil, %for.body50.epil ], [ %indvars.iv440.unr, %sw.epilog.loopexit479.unr-lcssa ]
  %epil.iter523 = phi i64 [ %epil.iter523.next, %for.body50.epil ], [ 0, %sw.epilog.loopexit479.unr-lcssa ]
  %indvars444.epil = trunc i64 %indvars.iv440.epil to i32
  %i45.0331.epil = add nsw i32 %indvars444.epil, -1
  %mul52330.epil = mul i32 %i45.0331.epil, %indvars444.epil
  %sub332.epil = add i32 %mul52330.epil, 3
  %add55.epil = mul i32 %sub332.epil, %indvars444.epil
  %arrayidx57.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv440.epil
  store i32 %add55.epil, ptr %arrayidx57.epil, align 4, !tbaa !4
  %indvars.iv.next441.epil = add nuw nsw i64 %indvars.iv440.epil, 1
  %epil.iter523.next = add i64 %epil.iter523, 1
  %epil.iter523.cmp.not = icmp eq i64 %epil.iter523.next, %xtraiter522
  br i1 %epil.iter523.cmp.not, label %sw.epilog, label %for.body50.epil, !llvm.loop !19

sw.epilog.loopexit480.unr-lcssa:                  ; preds = %for.body68, %for.body68.preheader
  %indvars.iv434.unr = phi i64 [ 0, %for.body68.preheader ], [ %indvars.iv.next435.3, %for.body68 ]
  %lcmp.mod519.not = icmp eq i64 %xtraiter517, 0
  br i1 %lcmp.mod519.not, label %sw.epilog, label %for.body68.epil

for.body68.epil:                                  ; preds = %sw.epilog.loopexit480.unr-lcssa, %for.body68.epil
  %indvars.iv434.epil = phi i64 [ %indvars.iv.next435.epil, %for.body68.epil ], [ %indvars.iv434.unr, %sw.epilog.loopexit480.unr-lcssa ]
  %epil.iter518 = phi i64 [ %epil.iter518.next, %for.body68.epil ], [ 0, %sw.epilog.loopexit480.unr-lcssa ]
  %indvars.iv.next435.epil = add nuw nsw i64 %indvars.iv434.epil, 1
  %indvars436.epil = trunc i64 %indvars.iv.next435.epil to i32
  %30 = trunc nuw nsw i64 %indvars.iv434.epil to i32
  %add70.epil = mul i32 %indvars436.epil, %30
  %add71.epil = add nsw i32 %add70.epil, 3
  %arrayidx73.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv434.epil
  store i32 %add71.epil, ptr %arrayidx73.epil, align 4, !tbaa !4
  %epil.iter518.next = add i64 %epil.iter518, 1
  %epil.iter518.cmp.not = icmp eq i64 %epil.iter518.next, %xtraiter517
  br i1 %epil.iter518.cmp.not, label %sw.epilog, label %for.body68.epil, !llvm.loop !20

sw.epilog.loopexit481.unr-lcssa:                  ; preds = %for.body84, %for.body84.preheader
  %indvars.iv426.unr = phi i64 [ 0, %for.body84.preheader ], [ %indvars.iv.next427.3, %for.body84 ]
  %lcmp.mod514.not = icmp eq i64 %xtraiter512, 0
  br i1 %lcmp.mod514.not, label %sw.epilog, label %for.body84.epil

for.body84.epil:                                  ; preds = %sw.epilog.loopexit481.unr-lcssa, %for.body84.epil
  %indvars.iv426.epil = phi i64 [ %indvars.iv.next427.epil, %for.body84.epil ], [ %indvars.iv426.unr, %sw.epilog.loopexit481.unr-lcssa ]
  %epil.iter513 = phi i64 [ %epil.iter513.next, %for.body84.epil ], [ 0, %sw.epilog.loopexit481.unr-lcssa ]
  %indvars431.epil = trunc i64 %indvars.iv426.epil to i32
  %mul85.epil = mul nuw nsw i32 %indvars431.epil, %indvars431.epil
  %mul86325.epil = add nuw i32 %mul85.epil, 1
  %mul87324.epil = mul i32 %mul86325.epil, %indvars431.epil
  %add89326.epil = add i32 %mul87324.epil, 3
  %add91.epil = mul i32 %add89326.epil, %indvars431.epil
  %arrayidx93.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv426.epil
  store i32 %add91.epil, ptr %arrayidx93.epil, align 4, !tbaa !4
  %indvars.iv.next427.epil = add nuw nsw i64 %indvars.iv426.epil, 1
  %epil.iter513.next = add i64 %epil.iter513, 1
  %epil.iter513.cmp.not = icmp eq i64 %epil.iter513.next, %xtraiter512
  br i1 %epil.iter513.cmp.not, label %sw.epilog, label %for.body84.epil, !llvm.loop !21

sw.epilog:                                        ; preds = %sw.epilog.loopexit481.unr-lcssa, %for.body84.epil, %sw.epilog.loopexit480.unr-lcssa, %for.body68.epil, %sw.epilog.loopexit479.unr-lcssa, %for.body50.epil, %sw.epilog.loopexit478.unr-lcssa, %for.body31.epil, %sw.epilog.loopexit.unr-lcssa, %for.body15.epil, %sw.bb77, %sw.bb61, %sw.bb43, %sw.bb24, %sw.bb, %if.then7
  tail call void @func4()
  unreachable

if.else:                                          ; preds = %if.then
  %puts306 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.19)
  %31 = load i32, ptr @len, align 4, !tbaa !4
  %cmp100352 = icmp sgt i32 %31, 0
  br i1 %cmp100352, label %for.body103, label %for.cond.cleanup102

for.cond.cleanup102:                              ; preds = %for.body103, %if.else
  %.lcssa = phi i32 [ %31, %if.else ], [ %33, %for.body103 ]
  %rem110 = srem i32 %.lcssa, 5
  switch i32 %rem110, label %sw.epilog199 [
    i32 0, label %sw.bb111
    i32 1, label %sw.bb131
    i32 2, label %sw.bb147
    i32 3, label %sw.bb164
    i32 4, label %sw.bb183
  ]

for.body103:                                      ; preds = %if.else, %for.body103
  %indvars.iv388 = phi i64 [ %indvars.iv.next389, %for.body103 ], [ 0, %if.else ]
  %arrayidx105 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv388
  %32 = load i32, ptr %arrayidx105, align 4, !tbaa !4
  %call106 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef signext %32)
  %indvars.iv.next389 = add nuw nsw i64 %indvars.iv388, 1
  %33 = load i32, ptr @len, align 4, !tbaa !4
  %34 = sext i32 %33 to i64
  %cmp100 = icmp slt i64 %indvars.iv.next389, %34
  br i1 %cmp100, label %for.body103, label %for.cond.cleanup102, !llvm.loop !22

sw.bb111:                                         ; preds = %for.cond.cleanup102
  %puts318 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %35 = load i32, ptr @len, align 4, !tbaa !4
  %cmp115362 = icmp sgt i32 %35, 0
  br i1 %cmp115362, label %for.body118.preheader, label %sw.epilog199

for.body118.preheader:                            ; preds = %sw.bb111
  %wide.trip.count424 = zext nneg i32 %35 to i64
  %xtraiter507 = and i64 %wide.trip.count424, 3
  %36 = icmp ult i32 %35, 4
  br i1 %36, label %sw.epilog199.loopexit.unr-lcssa, label %for.body118.preheader.new

for.body118.preheader.new:                        ; preds = %for.body118.preheader
  %unroll_iter510 = and i64 %wide.trip.count424, 2147483644
  br label %for.body118

for.body118:                                      ; preds = %for.body118, %for.body118.preheader.new
  %indvars.iv418 = phi i64 [ 0, %for.body118.preheader.new ], [ %indvars.iv.next419.3, %for.body118 ]
  %niter511 = phi i64 [ 0, %for.body118.preheader.new ], [ %niter511.next.3, %for.body118 ]
  %indvars423 = trunc i64 %indvars.iv418 to i32
  %mul119 = mul nuw nsw i32 %indvars423, %indvars423
  %mul120320 = or disjoint i32 %mul119, 1
  %mul121319 = mul i32 %mul120320, %indvars423
  %add123321 = or disjoint i32 %mul121319, 3
  %add125 = mul i32 %add123321, %indvars423
  %arrayidx127 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv418
  store i32 %add125, ptr %arrayidx127, align 8, !tbaa !4
  %indvars.iv.next419 = or disjoint i64 %indvars.iv418, 1
  %indvars423.1 = trunc i64 %indvars.iv.next419 to i32
  %mul119.1 = mul nuw nsw i32 %indvars423.1, %indvars423.1
  %mul120320.1 = add nuw nsw i32 %mul119.1, 1
  %mul121319.1 = mul i32 %mul120320.1, %indvars423.1
  %add123321.1 = add i32 %mul121319.1, 3
  %add125.1 = mul i32 %add123321.1, %indvars423.1
  %arrayidx127.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next419
  store i32 %add125.1, ptr %arrayidx127.1, align 4, !tbaa !4
  %indvars.iv.next419.1 = or disjoint i64 %indvars.iv418, 2
  %indvars423.2 = trunc i64 %indvars.iv.next419.1 to i32
  %mul119.2 = mul nuw nsw i32 %indvars423.2, %indvars423.2
  %mul120320.2 = or disjoint i32 %mul119.2, 1
  %mul121319.2 = mul i32 %mul120320.2, %indvars423.2
  %add123321.2 = add i32 %mul121319.2, 3
  %add125.2 = mul i32 %add123321.2, %indvars423.2
  %arrayidx127.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next419.1
  store i32 %add125.2, ptr %arrayidx127.2, align 8, !tbaa !4
  %indvars.iv.next419.2 = or disjoint i64 %indvars.iv418, 3
  %indvars423.3 = trunc i64 %indvars.iv.next419.2 to i32
  %mul119.3 = mul nuw nsw i32 %indvars423.3, %indvars423.3
  %mul120320.3 = add nuw nsw i32 %mul119.3, 1
  %mul121319.3 = mul i32 %mul120320.3, %indvars423.3
  %add123321.3 = add i32 %mul121319.3, 3
  %add125.3 = mul i32 %add123321.3, %indvars423.3
  %arrayidx127.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next419.2
  store i32 %add125.3, ptr %arrayidx127.3, align 4, !tbaa !4
  %indvars.iv.next419.3 = add nuw nsw i64 %indvars.iv418, 4
  %niter511.next.3 = add i64 %niter511, 4
  %niter511.ncmp.3 = icmp eq i64 %niter511.next.3, %unroll_iter510
  br i1 %niter511.ncmp.3, label %sw.epilog199.loopexit.unr-lcssa, label %for.body118, !llvm.loop !23

sw.bb131:                                         ; preds = %for.cond.cleanup102
  %puts316 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.29)
  %37 = load i32, ptr @len, align 4, !tbaa !4
  %cmp135360 = icmp sgt i32 %37, 0
  br i1 %cmp135360, label %for.body138.preheader, label %sw.epilog199

for.body138.preheader:                            ; preds = %sw.bb131
  %wide.trip.count416 = zext nneg i32 %37 to i64
  %xtraiter502 = and i64 %wide.trip.count416, 3
  %38 = icmp ult i32 %37, 4
  br i1 %38, label %sw.epilog199.loopexit482.unr-lcssa, label %for.body138.preheader.new

for.body138.preheader.new:                        ; preds = %for.body138.preheader
  %unroll_iter505 = and i64 %wide.trip.count416, 2147483644
  br label %for.body138

for.body138:                                      ; preds = %for.body138, %for.body138.preheader.new
  %indvars.iv412 = phi i64 [ 0, %for.body138.preheader.new ], [ %indvars.iv.next413.3, %for.body138 ]
  %niter506 = phi i64 [ 0, %for.body138.preheader.new ], [ %niter506.next.3, %for.body138 ]
  %indvars.iv.next413 = or disjoint i64 %indvars.iv412, 1
  %indvars414 = trunc i64 %indvars.iv.next413 to i32
  %39 = trunc nuw nsw i64 %indvars.iv412 to i32
  %add140 = mul i32 %indvars414, %39
  %add141 = or disjoint i32 %add140, 3
  %arrayidx143 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv412
  store i32 %add141, ptr %arrayidx143, align 8, !tbaa !4
  %indvars.iv.next413.1 = or disjoint i64 %indvars.iv412, 2
  %indvars414.1 = trunc i64 %indvars.iv.next413.1 to i32
  %40 = trunc nuw nsw i64 %indvars.iv.next413 to i32
  %add140.1 = mul i32 %indvars414.1, %40
  %add141.1 = add nsw i32 %add140.1, 3
  %arrayidx143.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next413
  store i32 %add141.1, ptr %arrayidx143.1, align 4, !tbaa !4
  %indvars.iv.next413.2 = or disjoint i64 %indvars.iv412, 3
  %indvars414.2 = trunc i64 %indvars.iv.next413.2 to i32
  %41 = trunc nuw nsw i64 %indvars.iv.next413.1 to i32
  %add140.2 = mul i32 %indvars414.2, %41
  %add141.2 = add nsw i32 %add140.2, 3
  %arrayidx143.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next413.1
  store i32 %add141.2, ptr %arrayidx143.2, align 8, !tbaa !4
  %indvars.iv.next413.3 = add nuw nsw i64 %indvars.iv412, 4
  %indvars414.3 = trunc i64 %indvars.iv.next413.3 to i32
  %42 = trunc nuw nsw i64 %indvars.iv.next413.2 to i32
  %add140.3 = mul i32 %indvars414.3, %42
  %add141.3 = or disjoint i32 %add140.3, 3
  %arrayidx143.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next413.2
  store i32 %add141.3, ptr %arrayidx143.3, align 4, !tbaa !4
  %niter506.next.3 = add i64 %niter506, 4
  %niter506.ncmp.3 = icmp eq i64 %niter506.next.3, %unroll_iter505
  br i1 %niter506.ncmp.3, label %sw.epilog199.loopexit482.unr-lcssa, label %for.body138, !llvm.loop !24

sw.bb147:                                         ; preds = %for.cond.cleanup102
  %puts313 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.28)
  %43 = load i32, ptr @len, align 4, !tbaa !4
  %cmp151358 = icmp sgt i32 %43, 0
  br i1 %cmp151358, label %for.body154.preheader, label %sw.epilog199

for.body154.preheader:                            ; preds = %sw.bb147
  %wide.trip.count410 = zext nneg i32 %43 to i64
  %xtraiter497 = and i64 %wide.trip.count410, 3
  %44 = icmp ult i32 %43, 4
  br i1 %44, label %sw.epilog199.loopexit483.unr-lcssa, label %for.body154.preheader.new

for.body154.preheader.new:                        ; preds = %for.body154.preheader
  %unroll_iter500 = and i64 %wide.trip.count410, 2147483644
  br label %for.body154

for.body154:                                      ; preds = %for.body154, %for.body154.preheader.new
  %indvars.iv405 = phi i64 [ 0, %for.body154.preheader.new ], [ %indvars.iv.next406.3, %for.body154 ]
  %niter501 = phi i64 [ 0, %for.body154.preheader.new ], [ %niter501.next.3, %for.body154 ]
  %45 = trunc nuw nsw i64 %indvars.iv405 to i32
  %mul156314 = mul i32 %45, %45
  %46 = trunc i64 %indvars.iv405 to i32
  %47 = add i32 %46, -1
  %sub158 = mul i32 %mul156314, %47
  %arrayidx160 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv405
  store i32 %sub158, ptr %arrayidx160, align 8, !tbaa !4
  %indvars.iv.next406 = or disjoint i64 %indvars.iv405, 1
  %48 = trunc nuw nsw i64 %indvars.iv.next406 to i32
  %mul156314.1 = mul i32 %48, %48
  %49 = trunc i64 %indvars.iv.next406 to i32
  %50 = add nsw i32 %49, -1
  %sub158.1 = mul i32 %mul156314.1, %50
  %arrayidx160.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next406
  store i32 %sub158.1, ptr %arrayidx160.1, align 4, !tbaa !4
  %indvars.iv.next406.1 = or disjoint i64 %indvars.iv405, 2
  %51 = trunc nuw nsw i64 %indvars.iv.next406.1 to i32
  %mul156314.2 = mul i32 %51, %51
  %52 = trunc i64 %indvars.iv.next406.1 to i32
  %53 = add nsw i32 %52, -1
  %sub158.2 = mul i32 %mul156314.2, %53
  %arrayidx160.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next406.1
  store i32 %sub158.2, ptr %arrayidx160.2, align 8, !tbaa !4
  %indvars.iv.next406.2 = or disjoint i64 %indvars.iv405, 3
  %54 = trunc nuw nsw i64 %indvars.iv.next406.2 to i32
  %mul156314.3 = mul i32 %54, %54
  %55 = trunc i64 %indvars.iv.next406.2 to i32
  %56 = add nsw i32 %55, -1
  %sub158.3 = mul i32 %mul156314.3, %56
  %arrayidx160.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next406.2
  store i32 %sub158.3, ptr %arrayidx160.3, align 4, !tbaa !4
  %indvars.iv.next406.3 = add nuw nsw i64 %indvars.iv405, 4
  %niter501.next.3 = add i64 %niter501, 4
  %niter501.ncmp.3 = icmp eq i64 %niter501.next.3, %unroll_iter500
  br i1 %niter501.ncmp.3, label %sw.epilog199.loopexit483.unr-lcssa, label %for.body154, !llvm.loop !25

sw.bb164:                                         ; preds = %for.cond.cleanup102
  %puts309 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.27)
  %57 = load i32, ptr @len, align 4, !tbaa !4
  %cmp168356 = icmp sgt i32 %57, 0
  br i1 %cmp168356, label %for.body171.preheader, label %sw.epilog199

for.body171.preheader:                            ; preds = %sw.bb164
  %wide.trip.count403 = zext nneg i32 %57 to i64
  %xtraiter492 = and i64 %wide.trip.count403, 3
  %58 = icmp ult i32 %57, 4
  br i1 %58, label %sw.epilog199.loopexit484.unr-lcssa, label %for.body171.preheader.new

for.body171.preheader.new:                        ; preds = %for.body171.preheader
  %unroll_iter495 = and i64 %wide.trip.count403, 2147483644
  br label %for.body171

for.body171:                                      ; preds = %for.body171, %for.body171.preheader.new
  %indvars.iv399 = phi i64 [ 0, %for.body171.preheader.new ], [ %indvars.iv.next400.3, %for.body171 ]
  %niter496 = phi i64 [ 0, %for.body171.preheader.new ], [ %niter496.next.3, %for.body171 ]
  %indvars.iv.next400 = or disjoint i64 %indvars.iv399, 1
  %indvars401 = trunc i64 %indvars.iv.next400 to i32
  %59 = trunc nuw nsw i64 %indvars.iv399 to i32
  %mul173310 = mul i32 %indvars401, %59
  %add175312 = or disjoint i32 %mul173310, 3
  %add177 = mul i32 %add175312, %59
  %arrayidx179 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv399
  store i32 %add177, ptr %arrayidx179, align 8, !tbaa !4
  %indvars.iv.next400.1 = or disjoint i64 %indvars.iv399, 2
  %indvars401.1 = trunc i64 %indvars.iv.next400.1 to i32
  %60 = trunc nuw nsw i64 %indvars.iv.next400 to i32
  %mul173310.1 = mul i32 %indvars401.1, %60
  %add175312.1 = add i32 %mul173310.1, 3
  %add177.1 = mul i32 %add175312.1, %60
  %arrayidx179.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next400
  store i32 %add177.1, ptr %arrayidx179.1, align 4, !tbaa !4
  %indvars.iv.next400.2 = or disjoint i64 %indvars.iv399, 3
  %indvars401.2 = trunc i64 %indvars.iv.next400.2 to i32
  %61 = trunc nuw nsw i64 %indvars.iv.next400.1 to i32
  %mul173310.2 = mul i32 %indvars401.2, %61
  %add175312.2 = add i32 %mul173310.2, 3
  %add177.2 = mul i32 %add175312.2, %61
  %arrayidx179.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next400.1
  store i32 %add177.2, ptr %arrayidx179.2, align 8, !tbaa !4
  %indvars.iv.next400.3 = add nuw nsw i64 %indvars.iv399, 4
  %indvars401.3 = trunc i64 %indvars.iv.next400.3 to i32
  %62 = trunc nuw nsw i64 %indvars.iv.next400.2 to i32
  %mul173310.3 = mul i32 %indvars401.3, %62
  %add175312.3 = or disjoint i32 %mul173310.3, 3
  %add177.3 = mul i32 %add175312.3, %62
  %arrayidx179.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next400.2
  store i32 %add177.3, ptr %arrayidx179.3, align 4, !tbaa !4
  %niter496.next.3 = add i64 %niter496, 4
  %niter496.ncmp.3 = icmp eq i64 %niter496.next.3, %unroll_iter495
  br i1 %niter496.ncmp.3, label %sw.epilog199.loopexit484.unr-lcssa, label %for.body171, !llvm.loop !26

sw.bb183:                                         ; preds = %for.cond.cleanup102
  %puts307 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.26)
  %63 = load i32, ptr @len, align 4, !tbaa !4
  %cmp187354 = icmp sgt i32 %63, 0
  br i1 %cmp187354, label %for.body190.preheader, label %sw.epilog199

for.body190.preheader:                            ; preds = %sw.bb183
  %wide.trip.count397 = zext nneg i32 %63 to i64
  %xtraiter487 = and i64 %wide.trip.count397, 3
  %64 = icmp ult i32 %63, 4
  br i1 %64, label %sw.epilog199.loopexit485.unr-lcssa, label %for.body190.preheader.new

for.body190.preheader.new:                        ; preds = %for.body190.preheader
  %unroll_iter490 = and i64 %wide.trip.count397, 2147483644
  br label %for.body190

for.body190:                                      ; preds = %for.body190, %for.body190.preheader.new
  %indvars.iv392 = phi i64 [ 0, %for.body190.preheader.new ], [ %indvars.iv.next393.3, %for.body190 ]
  %niter491 = phi i64 [ 0, %for.body190.preheader.new ], [ %niter491.next.3, %for.body190 ]
  %indvars396 = trunc i64 %indvars.iv392 to i32
  %mul192308 = or disjoint i32 %indvars396, 3
  %add193 = mul i32 %mul192308, %indvars396
  %arrayidx195 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv392
  store i32 %add193, ptr %arrayidx195, align 8, !tbaa !4
  %indvars.iv.next393 = or disjoint i64 %indvars.iv392, 1
  %indvars396.1 = trunc i64 %indvars.iv.next393 to i32
  %mul192308.1 = add nuw i32 %indvars396.1, 3
  %add193.1 = mul i32 %mul192308.1, %indvars396.1
  %arrayidx195.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next393
  store i32 %add193.1, ptr %arrayidx195.1, align 4, !tbaa !4
  %indvars.iv.next393.1 = or disjoint i64 %indvars.iv392, 2
  %indvars396.2 = trunc i64 %indvars.iv.next393.1 to i32
  %mul192308.2 = add nuw i32 %indvars396.2, 3
  %add193.2 = mul i32 %mul192308.2, %indvars396.2
  %arrayidx195.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next393.1
  store i32 %add193.2, ptr %arrayidx195.2, align 8, !tbaa !4
  %indvars.iv.next393.2 = or disjoint i64 %indvars.iv392, 3
  %indvars396.3 = trunc i64 %indvars.iv.next393.2 to i32
  %mul192308.3 = add nuw i32 %indvars396.3, 3
  %add193.3 = mul i32 %mul192308.3, %indvars396.3
  %arrayidx195.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next393.2
  store i32 %add193.3, ptr %arrayidx195.3, align 4, !tbaa !4
  %indvars.iv.next393.3 = add nuw nsw i64 %indvars.iv392, 4
  %niter491.next.3 = add i64 %niter491, 4
  %niter491.ncmp.3 = icmp eq i64 %niter491.next.3, %unroll_iter490
  br i1 %niter491.ncmp.3, label %sw.epilog199.loopexit485.unr-lcssa, label %for.body190, !llvm.loop !27

sw.epilog199.loopexit.unr-lcssa:                  ; preds = %for.body118, %for.body118.preheader
  %indvars.iv418.unr = phi i64 [ 0, %for.body118.preheader ], [ %indvars.iv.next419.3, %for.body118 ]
  %lcmp.mod509.not = icmp eq i64 %xtraiter507, 0
  br i1 %lcmp.mod509.not, label %sw.epilog199, label %for.body118.epil

for.body118.epil:                                 ; preds = %sw.epilog199.loopexit.unr-lcssa, %for.body118.epil
  %indvars.iv418.epil = phi i64 [ %indvars.iv.next419.epil, %for.body118.epil ], [ %indvars.iv418.unr, %sw.epilog199.loopexit.unr-lcssa ]
  %epil.iter508 = phi i64 [ %epil.iter508.next, %for.body118.epil ], [ 0, %sw.epilog199.loopexit.unr-lcssa ]
  %indvars423.epil = trunc i64 %indvars.iv418.epil to i32
  %mul119.epil = mul nuw nsw i32 %indvars423.epil, %indvars423.epil
  %mul120320.epil = add nuw i32 %mul119.epil, 1
  %mul121319.epil = mul i32 %mul120320.epil, %indvars423.epil
  %add123321.epil = add i32 %mul121319.epil, 3
  %add125.epil = mul i32 %add123321.epil, %indvars423.epil
  %arrayidx127.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv418.epil
  store i32 %add125.epil, ptr %arrayidx127.epil, align 4, !tbaa !4
  %indvars.iv.next419.epil = add nuw nsw i64 %indvars.iv418.epil, 1
  %epil.iter508.next = add i64 %epil.iter508, 1
  %epil.iter508.cmp.not = icmp eq i64 %epil.iter508.next, %xtraiter507
  br i1 %epil.iter508.cmp.not, label %sw.epilog199, label %for.body118.epil, !llvm.loop !28

sw.epilog199.loopexit482.unr-lcssa:               ; preds = %for.body138, %for.body138.preheader
  %indvars.iv412.unr = phi i64 [ 0, %for.body138.preheader ], [ %indvars.iv.next413.3, %for.body138 ]
  %lcmp.mod504.not = icmp eq i64 %xtraiter502, 0
  br i1 %lcmp.mod504.not, label %sw.epilog199, label %for.body138.epil

for.body138.epil:                                 ; preds = %sw.epilog199.loopexit482.unr-lcssa, %for.body138.epil
  %indvars.iv412.epil = phi i64 [ %indvars.iv.next413.epil, %for.body138.epil ], [ %indvars.iv412.unr, %sw.epilog199.loopexit482.unr-lcssa ]
  %epil.iter503 = phi i64 [ %epil.iter503.next, %for.body138.epil ], [ 0, %sw.epilog199.loopexit482.unr-lcssa ]
  %indvars.iv.next413.epil = add nuw nsw i64 %indvars.iv412.epil, 1
  %indvars414.epil = trunc i64 %indvars.iv.next413.epil to i32
  %65 = trunc nuw nsw i64 %indvars.iv412.epil to i32
  %add140.epil = mul i32 %indvars414.epil, %65
  %add141.epil = add nsw i32 %add140.epil, 3
  %arrayidx143.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv412.epil
  store i32 %add141.epil, ptr %arrayidx143.epil, align 4, !tbaa !4
  %epil.iter503.next = add i64 %epil.iter503, 1
  %epil.iter503.cmp.not = icmp eq i64 %epil.iter503.next, %xtraiter502
  br i1 %epil.iter503.cmp.not, label %sw.epilog199, label %for.body138.epil, !llvm.loop !29

sw.epilog199.loopexit483.unr-lcssa:               ; preds = %for.body154, %for.body154.preheader
  %indvars.iv405.unr = phi i64 [ 0, %for.body154.preheader ], [ %indvars.iv.next406.3, %for.body154 ]
  %lcmp.mod499.not = icmp eq i64 %xtraiter497, 0
  br i1 %lcmp.mod499.not, label %sw.epilog199, label %for.body154.epil

for.body154.epil:                                 ; preds = %sw.epilog199.loopexit483.unr-lcssa, %for.body154.epil
  %indvars.iv405.epil = phi i64 [ %indvars.iv.next406.epil, %for.body154.epil ], [ %indvars.iv405.unr, %sw.epilog199.loopexit483.unr-lcssa ]
  %epil.iter498 = phi i64 [ %epil.iter498.next, %for.body154.epil ], [ 0, %sw.epilog199.loopexit483.unr-lcssa ]
  %66 = trunc nuw nsw i64 %indvars.iv405.epil to i32
  %mul156314.epil = mul i32 %66, %66
  %67 = trunc i64 %indvars.iv405.epil to i32
  %68 = add i32 %67, -1
  %sub158.epil = mul i32 %mul156314.epil, %68
  %arrayidx160.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv405.epil
  store i32 %sub158.epil, ptr %arrayidx160.epil, align 4, !tbaa !4
  %indvars.iv.next406.epil = add nuw nsw i64 %indvars.iv405.epil, 1
  %epil.iter498.next = add i64 %epil.iter498, 1
  %epil.iter498.cmp.not = icmp eq i64 %epil.iter498.next, %xtraiter497
  br i1 %epil.iter498.cmp.not, label %sw.epilog199, label %for.body154.epil, !llvm.loop !30

sw.epilog199.loopexit484.unr-lcssa:               ; preds = %for.body171, %for.body171.preheader
  %indvars.iv399.unr = phi i64 [ 0, %for.body171.preheader ], [ %indvars.iv.next400.3, %for.body171 ]
  %lcmp.mod494.not = icmp eq i64 %xtraiter492, 0
  br i1 %lcmp.mod494.not, label %sw.epilog199, label %for.body171.epil

for.body171.epil:                                 ; preds = %sw.epilog199.loopexit484.unr-lcssa, %for.body171.epil
  %indvars.iv399.epil = phi i64 [ %indvars.iv.next400.epil, %for.body171.epil ], [ %indvars.iv399.unr, %sw.epilog199.loopexit484.unr-lcssa ]
  %epil.iter493 = phi i64 [ %epil.iter493.next, %for.body171.epil ], [ 0, %sw.epilog199.loopexit484.unr-lcssa ]
  %indvars.iv.next400.epil = add nuw nsw i64 %indvars.iv399.epil, 1
  %indvars401.epil = trunc i64 %indvars.iv.next400.epil to i32
  %69 = trunc nuw nsw i64 %indvars.iv399.epil to i32
  %mul173310.epil = mul i32 %indvars401.epil, %69
  %add175312.epil = add i32 %mul173310.epil, 3
  %add177.epil = mul i32 %add175312.epil, %69
  %arrayidx179.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv399.epil
  store i32 %add177.epil, ptr %arrayidx179.epil, align 4, !tbaa !4
  %epil.iter493.next = add i64 %epil.iter493, 1
  %epil.iter493.cmp.not = icmp eq i64 %epil.iter493.next, %xtraiter492
  br i1 %epil.iter493.cmp.not, label %sw.epilog199, label %for.body171.epil, !llvm.loop !31

sw.epilog199.loopexit485.unr-lcssa:               ; preds = %for.body190, %for.body190.preheader
  %indvars.iv392.unr = phi i64 [ 0, %for.body190.preheader ], [ %indvars.iv.next393.3, %for.body190 ]
  %lcmp.mod489.not = icmp eq i64 %xtraiter487, 0
  br i1 %lcmp.mod489.not, label %sw.epilog199, label %for.body190.epil

for.body190.epil:                                 ; preds = %sw.epilog199.loopexit485.unr-lcssa, %for.body190.epil
  %indvars.iv392.epil = phi i64 [ %indvars.iv.next393.epil, %for.body190.epil ], [ %indvars.iv392.unr, %sw.epilog199.loopexit485.unr-lcssa ]
  %epil.iter488 = phi i64 [ %epil.iter488.next, %for.body190.epil ], [ 0, %sw.epilog199.loopexit485.unr-lcssa ]
  %indvars396.epil = trunc i64 %indvars.iv392.epil to i32
  %mul192308.epil = add nuw i32 %indvars396.epil, 3
  %add193.epil = mul i32 %mul192308.epil, %indvars396.epil
  %arrayidx195.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv392.epil
  store i32 %add193.epil, ptr %arrayidx195.epil, align 4, !tbaa !4
  %indvars.iv.next393.epil = add nuw nsw i64 %indvars.iv392.epil, 1
  %epil.iter488.next = add i64 %epil.iter488, 1
  %epil.iter488.cmp.not = icmp eq i64 %epil.iter488.next, %xtraiter487
  br i1 %epil.iter488.cmp.not, label %sw.epilog199, label %for.body190.epil, !llvm.loop !32

sw.epilog199:                                     ; preds = %sw.epilog199.loopexit485.unr-lcssa, %for.body190.epil, %sw.epilog199.loopexit484.unr-lcssa, %for.body171.epil, %sw.epilog199.loopexit483.unr-lcssa, %for.body154.epil, %sw.epilog199.loopexit482.unr-lcssa, %for.body138.epil, %sw.epilog199.loopexit.unr-lcssa, %for.body118.epil, %sw.bb183, %sw.bb164, %sw.bb147, %sw.bb131, %sw.bb111, %for.cond.cleanup102
  tail call void @func3()
  unreachable

if.else200:                                       ; preds = %for.cond.cleanup
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.17)
  %70 = load i32, ptr @len, align 4, !tbaa !4
  %cmp204350 = icmp sgt i32 %70, 0
  br i1 %cmp204350, label %for.body207, label %for.cond.cleanup206

for.cond.cleanup206:                              ; preds = %for.body207, %if.else200
  tail call void @func2()
  unreachable

for.body207:                                      ; preds = %if.else200, %for.body207
  %indvars.iv384 = phi i64 [ %indvars.iv.next385, %for.body207 ], [ 0, %if.else200 ]
  %arrayidx209 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv384
  %71 = load i32, ptr %arrayidx209, align 4, !tbaa !4
  %call210 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef signext %71)
  %indvars.iv.next385 = add nuw nsw i64 %indvars.iv384, 1
  %72 = load i32, ptr @len, align 4, !tbaa !4
  %73 = sext i32 %72 to i64
  %cmp204 = icmp slt i64 %indvars.iv.next385, %73
  br i1 %cmp204, label %for.body207, label %for.cond.cleanup206, !llvm.loop !33
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
  %puts3 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.32)
  %call1 = tail call signext i32 @func1()
  unreachable

if.else:                                          ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.31)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #6

attributes #0 = { noinline noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { nofree nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #2 = { noreturn nounwind }
attributes #3 = { noreturn nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #4 = { nounwind }
attributes #5 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #6 = { nofree nounwind }

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
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.unroll.disable"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
!12 = distinct !{!12, !11}
!13 = distinct !{!13, !11}
!14 = distinct !{!14, !11}
!15 = distinct !{!15, !11}
!16 = distinct !{!16, !11}
!17 = distinct !{!17, !9}
!18 = distinct !{!18, !9}
!19 = distinct !{!19, !9}
!20 = distinct !{!20, !9}
!21 = distinct !{!21, !9}
!22 = distinct !{!22, !11}
!23 = distinct !{!23, !11}
!24 = distinct !{!24, !11}
!25 = distinct !{!25, !11}
!26 = distinct !{!26, !11}
!27 = distinct !{!27, !11}
!28 = distinct !{!28, !9}
!29 = distinct !{!29, !9}
!30 = distinct !{!30, !9}
!31 = distinct !{!31, !9}
!32 = distinct !{!32, !9}
!33 = distinct !{!33, !11}
