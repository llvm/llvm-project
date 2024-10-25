; -mbackchain option
; Test for Frame Pointer in first slot in jmp_buf.
; Test assembly for nested setjmp for alloa with switch statement.
; This test case takes input from stdin for size of alloca
; and produce the right result.
; Frame Pointer in slot 1.
; Return address in slot 2.
; Backchain value in slot 3.
; Stack Pointer in slot 4.

; RUN: llc < %s | FileCheck %s

; ModuleID = 'builtin-setjmp-longjmp-alloca-03.c'
source_filename = "builtin-setjmp-longjmp-alloca-03.c"
target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "s390x-unknown-linux-gnu"

@buf3 = dso_local global [10 x ptr] zeroinitializer, align 8
@buf2 = dso_local global [10 x ptr] zeroinitializer, align 8
@buf1 = dso_local global [10 x ptr] zeroinitializer, align 8
@.str.3 = private unnamed_addr constant [22 x i8] c"Please enter length: \00", align 2
@.str.4 = private unnamed_addr constant [3 x i8] c"%d\00", align 2
@.str.13 = private unnamed_addr constant [9 x i8] c"arr: %d\0A\00", align 2
@str = private unnamed_addr constant [9 x i8] c"In func4\00", align 1
@str.17 = private unnamed_addr constant [9 x i8] c"In func3\00", align 1
@str.18 = private unnamed_addr constant [9 x i8] c"In func2\00", align 1
@str.19 = private unnamed_addr constant [20 x i8] c"Returned from func3\00", align 1
@str.20 = private unnamed_addr constant [32 x i8] c"First __builtin_setjmp in func1\00", align 1
@str.21 = private unnamed_addr constant [20 x i8] c"Returned from func4\00", align 1
@str.27 = private unnamed_addr constant [33 x i8] c"Second __builtin_setjmp in func1\00", align 1
@str.28 = private unnamed_addr constant [8 x i8] c"case 4:\00", align 1
@str.29 = private unnamed_addr constant [8 x i8] c"case 3:\00", align 1
@str.30 = private unnamed_addr constant [8 x i8] c"case 2:\00", align 1
@str.31 = private unnamed_addr constant [8 x i8] c"case 1:\00", align 1
@str.32 = private unnamed_addr constant [8 x i8] c"case 0:\00", align 1
@str.33 = private unnamed_addr constant [44 x i8] c"In main, after __builtin_longjmp from func1\00", align 1
@str.34 = private unnamed_addr constant [20 x i8] c"In main, first time\00", align 1

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
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.17)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf2)
  unreachable
}

; Function Attrs: noinline noreturn nounwind
define dso_local void @func2() local_unnamed_addr #0 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.18)
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
  %cmp350 = icmp sgt i32 %0, 0
  br i1 %cmp350, label %for.body.preheader, label %for.cond.cleanup

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
  %4 = call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf2)
  %cmp4 = icmp eq i32 %4, 0
  br i1 %cmp4, label %if.then, label %if.else202

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
  %puts307 = call i32 @puts(ptr nonnull dereferenceable(1) @str.20)
  %9 = call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf3)
  %cmp7 = icmp eq i32 %9, 0
  br i1 %cmp7, label %if.then9, label %if.else

if.then9:                                         ; preds = %if.then
  %puts324 = call i32 @puts(ptr nonnull dereferenceable(1) @str.27)
  %10 = load i32, ptr %len, align 4, !tbaa !4
  %rem = srem i32 %10, 5
  switch i32 %rem, label %sw.epilog [
    i32 0, label %sw.bb
    i32 1, label %sw.bb26
    i32 2, label %sw.bb45
    i32 3, label %sw.bb63
    i32 4, label %sw.bb79
  ]

sw.bb:                                            ; preds = %if.then9
  %puts339 = call i32 @puts(ptr nonnull dereferenceable(1) @str.32)
  %11 = load i32, ptr %len, align 4, !tbaa !4
  %cmp14374 = icmp sgt i32 %11, 0
  br i1 %cmp14374, label %for.body17.preheader, label %sw.epilog

for.body17.preheader:                             ; preds = %sw.bb
  %wide.trip.count460 = zext nneg i32 %11 to i64
  %xtraiter534 = and i64 %wide.trip.count460, 3
  %12 = icmp ult i32 %11, 4
  br i1 %12, label %sw.epilog.loopexit.unr-lcssa, label %for.body17.preheader.new

for.body17.preheader.new:                         ; preds = %for.body17.preheader
  %unroll_iter537 = and i64 %wide.trip.count460, 2147483644
  br label %for.body17

for.body17:                                       ; preds = %for.body17, %for.body17.preheader.new
  %indvars.iv455 = phi i64 [ 0, %for.body17.preheader.new ], [ %indvars.iv.next456.3, %for.body17 ]
  %niter538 = phi i64 [ 0, %for.body17.preheader.new ], [ %niter538.next.3, %for.body17 ]
  %indvars459 = trunc i64 %indvars.iv455 to i32
  %mul19340 = or disjoint i32 %indvars459, 3
  %add20 = mul i32 %mul19340, %indvars459
  %arrayidx22 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv455
  store i32 %add20, ptr %arrayidx22, align 8, !tbaa !4
  %indvars.iv.next456 = or disjoint i64 %indvars.iv455, 1
  %indvars459.1 = trunc i64 %indvars.iv.next456 to i32
  %mul19340.1 = add nuw i32 %indvars459.1, 3
  %add20.1 = mul i32 %mul19340.1, %indvars459.1
  %arrayidx22.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next456
  store i32 %add20.1, ptr %arrayidx22.1, align 4, !tbaa !4
  %indvars.iv.next456.1 = or disjoint i64 %indvars.iv455, 2
  %indvars459.2 = trunc i64 %indvars.iv.next456.1 to i32
  %mul19340.2 = add nuw i32 %indvars459.2, 3
  %add20.2 = mul i32 %mul19340.2, %indvars459.2
  %arrayidx22.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next456.1
  store i32 %add20.2, ptr %arrayidx22.2, align 8, !tbaa !4
  %indvars.iv.next456.2 = or disjoint i64 %indvars.iv455, 3
  %indvars459.3 = trunc i64 %indvars.iv.next456.2 to i32
  %mul19340.3 = add nuw i32 %indvars459.3, 3
  %add20.3 = mul i32 %mul19340.3, %indvars459.3
  %arrayidx22.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next456.2
  store i32 %add20.3, ptr %arrayidx22.3, align 4, !tbaa !4
  %indvars.iv.next456.3 = add nuw nsw i64 %indvars.iv455, 4
  %niter538.next.3 = add i64 %niter538, 4
  %niter538.ncmp.3 = icmp eq i64 %niter538.next.3, %unroll_iter537
  br i1 %niter538.ncmp.3, label %sw.epilog.loopexit.unr-lcssa, label %for.body17, !llvm.loop !12

sw.bb26:                                          ; preds = %if.then9
  %puts335 = call i32 @puts(ptr nonnull dereferenceable(1) @str.31)
  %13 = load i32, ptr %len, align 4, !tbaa !4
  %cmp30372 = icmp sgt i32 %13, 0
  br i1 %cmp30372, label %for.body33.preheader, label %sw.epilog

for.body33.preheader:                             ; preds = %sw.bb26
  %wide.trip.count453 = zext nneg i32 %13 to i64
  %xtraiter529 = and i64 %wide.trip.count453, 3
  %14 = icmp ult i32 %13, 4
  br i1 %14, label %sw.epilog.loopexit480.unr-lcssa, label %for.body33.preheader.new

for.body33.preheader.new:                         ; preds = %for.body33.preheader
  %unroll_iter532 = and i64 %wide.trip.count453, 2147483644
  br label %for.body33

for.body33:                                       ; preds = %for.body33, %for.body33.preheader.new
  %indvars.iv449 = phi i64 [ 0, %for.body33.preheader.new ], [ %indvars.iv.next450.3, %for.body33 ]
  %niter533 = phi i64 [ 0, %for.body33.preheader.new ], [ %niter533.next.3, %for.body33 ]
  %indvars.iv.next450 = or disjoint i64 %indvars.iv449, 1
  %indvars451 = trunc i64 %indvars.iv.next450 to i32
  %15 = trunc nuw nsw i64 %indvars.iv449 to i32
  %mul35336 = mul i32 %indvars451, %15
  %add37338 = or disjoint i32 %mul35336, 3
  %add39 = mul i32 %add37338, %15
  %arrayidx41 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv449
  store i32 %add39, ptr %arrayidx41, align 8, !tbaa !4
  %indvars.iv.next450.1 = or disjoint i64 %indvars.iv449, 2
  %indvars451.1 = trunc i64 %indvars.iv.next450.1 to i32
  %16 = trunc nuw nsw i64 %indvars.iv.next450 to i32
  %mul35336.1 = mul i32 %indvars451.1, %16
  %add37338.1 = add i32 %mul35336.1, 3
  %add39.1 = mul i32 %add37338.1, %16
  %arrayidx41.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next450
  store i32 %add39.1, ptr %arrayidx41.1, align 4, !tbaa !4
  %indvars.iv.next450.2 = or disjoint i64 %indvars.iv449, 3
  %indvars451.2 = trunc i64 %indvars.iv.next450.2 to i32
  %17 = trunc nuw nsw i64 %indvars.iv.next450.1 to i32
  %mul35336.2 = mul i32 %indvars451.2, %17
  %add37338.2 = add i32 %mul35336.2, 3
  %add39.2 = mul i32 %add37338.2, %17
  %arrayidx41.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next450.1
  store i32 %add39.2, ptr %arrayidx41.2, align 8, !tbaa !4
  %indvars.iv.next450.3 = add nuw nsw i64 %indvars.iv449, 4
  %indvars451.3 = trunc i64 %indvars.iv.next450.3 to i32
  %18 = trunc nuw nsw i64 %indvars.iv.next450.2 to i32
  %mul35336.3 = mul i32 %indvars451.3, %18
  %add37338.3 = or disjoint i32 %mul35336.3, 3
  %add39.3 = mul i32 %add37338.3, %18
  %arrayidx41.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next450.2
  store i32 %add39.3, ptr %arrayidx41.3, align 4, !tbaa !4
  %niter533.next.3 = add i64 %niter533, 4
  %niter533.ncmp.3 = icmp eq i64 %niter533.next.3, %unroll_iter532
  br i1 %niter533.ncmp.3, label %sw.epilog.loopexit480.unr-lcssa, label %for.body33, !llvm.loop !13

sw.bb45:                                          ; preds = %if.then9
  %puts331 = call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %19 = load i32, ptr %len, align 4, !tbaa !4
  %cmp49370 = icmp sgt i32 %19, 0
  br i1 %cmp49370, label %for.body52.preheader, label %sw.epilog

for.body52.preheader:                             ; preds = %sw.bb45
  %wide.trip.count447 = zext nneg i32 %19 to i64
  %xtraiter524 = and i64 %wide.trip.count447, 3
  %20 = icmp ult i32 %19, 4
  br i1 %20, label %sw.epilog.loopexit481.unr-lcssa, label %for.body52.preheader.new

for.body52.preheader.new:                         ; preds = %for.body52.preheader
  %unroll_iter527 = and i64 %wide.trip.count447, 2147483644
  br label %for.body52

for.body52:                                       ; preds = %for.body52, %for.body52.preheader.new
  %indvars.iv442 = phi i64 [ 0, %for.body52.preheader.new ], [ %indvars.iv.next443.3, %for.body52 ]
  %niter528 = phi i64 [ 0, %for.body52.preheader.new ], [ %niter528.next.3, %for.body52 ]
  %indvars446 = trunc i64 %indvars.iv442 to i32
  %i47.0333 = add nsw i32 %indvars446, -1
  %mul54332 = mul i32 %i47.0333, %indvars446
  %sub334 = or disjoint i32 %mul54332, 3
  %add57 = mul i32 %sub334, %indvars446
  %arrayidx59 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv442
  store i32 %add57, ptr %arrayidx59, align 8, !tbaa !4
  %indvars.iv.next443 = or disjoint i64 %indvars.iv442, 1
  %indvars446.1 = trunc i64 %indvars.iv.next443 to i32
  %i47.0333.1 = add nsw i32 %indvars446.1, -1
  %mul54332.1 = mul i32 %i47.0333.1, %indvars446.1
  %sub334.1 = or disjoint i32 %mul54332.1, 3
  %add57.1 = mul i32 %sub334.1, %indvars446.1
  %arrayidx59.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next443
  store i32 %add57.1, ptr %arrayidx59.1, align 4, !tbaa !4
  %indvars.iv.next443.1 = or disjoint i64 %indvars.iv442, 2
  %indvars446.2 = trunc i64 %indvars.iv.next443.1 to i32
  %i47.0333.2 = add nsw i32 %indvars446.2, -1
  %mul54332.2 = mul i32 %i47.0333.2, %indvars446.2
  %sub334.2 = add i32 %mul54332.2, 3
  %add57.2 = mul i32 %sub334.2, %indvars446.2
  %arrayidx59.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next443.1
  store i32 %add57.2, ptr %arrayidx59.2, align 8, !tbaa !4
  %indvars.iv.next443.2 = or disjoint i64 %indvars.iv442, 3
  %indvars446.3 = trunc i64 %indvars.iv.next443.2 to i32
  %i47.0333.3 = add nsw i32 %indvars446.3, -1
  %mul54332.3 = mul i32 %i47.0333.3, %indvars446.3
  %sub334.3 = add i32 %mul54332.3, 3
  %add57.3 = mul i32 %sub334.3, %indvars446.3
  %arrayidx59.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next443.2
  store i32 %add57.3, ptr %arrayidx59.3, align 4, !tbaa !4
  %indvars.iv.next443.3 = add nuw nsw i64 %indvars.iv442, 4
  %niter528.next.3 = add i64 %niter528, 4
  %niter528.ncmp.3 = icmp eq i64 %niter528.next.3, %unroll_iter527
  br i1 %niter528.ncmp.3, label %sw.epilog.loopexit481.unr-lcssa, label %for.body52, !llvm.loop !14

sw.bb63:                                          ; preds = %if.then9
  %puts329 = call i32 @puts(ptr nonnull dereferenceable(1) @str.29)
  %21 = load i32, ptr %len, align 4, !tbaa !4
  %cmp67368 = icmp sgt i32 %21, 0
  br i1 %cmp67368, label %for.body70.preheader, label %sw.epilog

for.body70.preheader:                             ; preds = %sw.bb63
  %wide.trip.count440 = zext nneg i32 %21 to i64
  %xtraiter519 = and i64 %wide.trip.count440, 3
  %22 = icmp ult i32 %21, 4
  br i1 %22, label %sw.epilog.loopexit482.unr-lcssa, label %for.body70.preheader.new

for.body70.preheader.new:                         ; preds = %for.body70.preheader
  %unroll_iter522 = and i64 %wide.trip.count440, 2147483644
  br label %for.body70

for.body70:                                       ; preds = %for.body70, %for.body70.preheader.new
  %indvars.iv436 = phi i64 [ 0, %for.body70.preheader.new ], [ %indvars.iv.next437.3, %for.body70 ]
  %niter523 = phi i64 [ 0, %for.body70.preheader.new ], [ %niter523.next.3, %for.body70 ]
  %indvars.iv.next437 = or disjoint i64 %indvars.iv436, 1
  %indvars438 = trunc i64 %indvars.iv.next437 to i32
  %23 = trunc nuw nsw i64 %indvars.iv436 to i32
  %add72 = mul i32 %indvars438, %23
  %add73 = or disjoint i32 %add72, 3
  %arrayidx75 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv436
  store i32 %add73, ptr %arrayidx75, align 8, !tbaa !4
  %indvars.iv.next437.1 = or disjoint i64 %indvars.iv436, 2
  %indvars438.1 = trunc i64 %indvars.iv.next437.1 to i32
  %24 = trunc nuw nsw i64 %indvars.iv.next437 to i32
  %add72.1 = mul i32 %indvars438.1, %24
  %add73.1 = add nsw i32 %add72.1, 3
  %arrayidx75.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next437
  store i32 %add73.1, ptr %arrayidx75.1, align 4, !tbaa !4
  %indvars.iv.next437.2 = or disjoint i64 %indvars.iv436, 3
  %indvars438.2 = trunc i64 %indvars.iv.next437.2 to i32
  %25 = trunc nuw nsw i64 %indvars.iv.next437.1 to i32
  %add72.2 = mul i32 %indvars438.2, %25
  %add73.2 = add nsw i32 %add72.2, 3
  %arrayidx75.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next437.1
  store i32 %add73.2, ptr %arrayidx75.2, align 8, !tbaa !4
  %indvars.iv.next437.3 = add nuw nsw i64 %indvars.iv436, 4
  %indvars438.3 = trunc i64 %indvars.iv.next437.3 to i32
  %26 = trunc nuw nsw i64 %indvars.iv.next437.2 to i32
  %add72.3 = mul i32 %indvars438.3, %26
  %add73.3 = or disjoint i32 %add72.3, 3
  %arrayidx75.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next437.2
  store i32 %add73.3, ptr %arrayidx75.3, align 4, !tbaa !4
  %niter523.next.3 = add i64 %niter523, 4
  %niter523.ncmp.3 = icmp eq i64 %niter523.next.3, %unroll_iter522
  br i1 %niter523.ncmp.3, label %sw.epilog.loopexit482.unr-lcssa, label %for.body70, !llvm.loop !15

sw.bb79:                                          ; preds = %if.then9
  %puts325 = call i32 @puts(ptr nonnull dereferenceable(1) @str.28)
  %27 = load i32, ptr %len, align 4, !tbaa !4
  %cmp83366 = icmp sgt i32 %27, 0
  br i1 %cmp83366, label %for.body86.preheader, label %sw.epilog

for.body86.preheader:                             ; preds = %sw.bb79
  %wide.trip.count434 = zext nneg i32 %27 to i64
  %xtraiter514 = and i64 %wide.trip.count434, 3
  %28 = icmp ult i32 %27, 4
  br i1 %28, label %sw.epilog.loopexit483.unr-lcssa, label %for.body86.preheader.new

for.body86.preheader.new:                         ; preds = %for.body86.preheader
  %unroll_iter517 = and i64 %wide.trip.count434, 2147483644
  br label %for.body86

for.body86:                                       ; preds = %for.body86, %for.body86.preheader.new
  %indvars.iv428 = phi i64 [ 0, %for.body86.preheader.new ], [ %indvars.iv.next429.3, %for.body86 ]
  %niter518 = phi i64 [ 0, %for.body86.preheader.new ], [ %niter518.next.3, %for.body86 ]
  %indvars433 = trunc i64 %indvars.iv428 to i32
  %mul87 = mul nuw nsw i32 %indvars433, %indvars433
  %mul88327 = or disjoint i32 %mul87, 1
  %mul89326 = mul i32 %mul88327, %indvars433
  %add91328 = or disjoint i32 %mul89326, 3
  %add93 = mul i32 %add91328, %indvars433
  %arrayidx95 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv428
  store i32 %add93, ptr %arrayidx95, align 8, !tbaa !4
  %indvars.iv.next429 = or disjoint i64 %indvars.iv428, 1
  %indvars433.1 = trunc i64 %indvars.iv.next429 to i32
  %mul87.1 = mul nuw nsw i32 %indvars433.1, %indvars433.1
  %mul88327.1 = add nuw nsw i32 %mul87.1, 1
  %mul89326.1 = mul i32 %mul88327.1, %indvars433.1
  %add91328.1 = add i32 %mul89326.1, 3
  %add93.1 = mul i32 %add91328.1, %indvars433.1
  %arrayidx95.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next429
  store i32 %add93.1, ptr %arrayidx95.1, align 4, !tbaa !4
  %indvars.iv.next429.1 = or disjoint i64 %indvars.iv428, 2
  %indvars433.2 = trunc i64 %indvars.iv.next429.1 to i32
  %mul87.2 = mul nuw nsw i32 %indvars433.2, %indvars433.2
  %mul88327.2 = or disjoint i32 %mul87.2, 1
  %mul89326.2 = mul i32 %mul88327.2, %indvars433.2
  %add91328.2 = add i32 %mul89326.2, 3
  %add93.2 = mul i32 %add91328.2, %indvars433.2
  %arrayidx95.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next429.1
  store i32 %add93.2, ptr %arrayidx95.2, align 8, !tbaa !4
  %indvars.iv.next429.2 = or disjoint i64 %indvars.iv428, 3
  %indvars433.3 = trunc i64 %indvars.iv.next429.2 to i32
  %mul87.3 = mul nuw nsw i32 %indvars433.3, %indvars433.3
  %mul88327.3 = add nuw nsw i32 %mul87.3, 1
  %mul89326.3 = mul i32 %mul88327.3, %indvars433.3
  %add91328.3 = add i32 %mul89326.3, 3
  %add93.3 = mul i32 %add91328.3, %indvars433.3
  %arrayidx95.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next429.2
  store i32 %add93.3, ptr %arrayidx95.3, align 4, !tbaa !4
  %indvars.iv.next429.3 = add nuw nsw i64 %indvars.iv428, 4
  %niter518.next.3 = add i64 %niter518, 4
  %niter518.ncmp.3 = icmp eq i64 %niter518.next.3, %unroll_iter517
  br i1 %niter518.ncmp.3, label %sw.epilog.loopexit483.unr-lcssa, label %for.body86, !llvm.loop !16

sw.epilog.loopexit.unr-lcssa:                     ; preds = %for.body17, %for.body17.preheader
  %indvars.iv455.unr = phi i64 [ 0, %for.body17.preheader ], [ %indvars.iv.next456.3, %for.body17 ]
  %lcmp.mod536.not = icmp eq i64 %xtraiter534, 0
  br i1 %lcmp.mod536.not, label %sw.epilog, label %for.body17.epil

for.body17.epil:                                  ; preds = %sw.epilog.loopexit.unr-lcssa, %for.body17.epil
  %indvars.iv455.epil = phi i64 [ %indvars.iv.next456.epil, %for.body17.epil ], [ %indvars.iv455.unr, %sw.epilog.loopexit.unr-lcssa ]
  %epil.iter535 = phi i64 [ %epil.iter535.next, %for.body17.epil ], [ 0, %sw.epilog.loopexit.unr-lcssa ]
  %indvars459.epil = trunc i64 %indvars.iv455.epil to i32
  %mul19340.epil = add nuw i32 %indvars459.epil, 3
  %add20.epil = mul i32 %mul19340.epil, %indvars459.epil
  %arrayidx22.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv455.epil
  store i32 %add20.epil, ptr %arrayidx22.epil, align 4, !tbaa !4
  %indvars.iv.next456.epil = add nuw nsw i64 %indvars.iv455.epil, 1
  %epil.iter535.next = add i64 %epil.iter535, 1
  %epil.iter535.cmp.not = icmp eq i64 %epil.iter535.next, %xtraiter534
  br i1 %epil.iter535.cmp.not, label %sw.epilog, label %for.body17.epil, !llvm.loop !17

sw.epilog.loopexit480.unr-lcssa:                  ; preds = %for.body33, %for.body33.preheader
  %indvars.iv449.unr = phi i64 [ 0, %for.body33.preheader ], [ %indvars.iv.next450.3, %for.body33 ]
  %lcmp.mod531.not = icmp eq i64 %xtraiter529, 0
  br i1 %lcmp.mod531.not, label %sw.epilog, label %for.body33.epil

for.body33.epil:                                  ; preds = %sw.epilog.loopexit480.unr-lcssa, %for.body33.epil
  %indvars.iv449.epil = phi i64 [ %indvars.iv.next450.epil, %for.body33.epil ], [ %indvars.iv449.unr, %sw.epilog.loopexit480.unr-lcssa ]
  %epil.iter530 = phi i64 [ %epil.iter530.next, %for.body33.epil ], [ 0, %sw.epilog.loopexit480.unr-lcssa ]
  %indvars.iv.next450.epil = add nuw nsw i64 %indvars.iv449.epil, 1
  %indvars451.epil = trunc i64 %indvars.iv.next450.epil to i32
  %29 = trunc nuw nsw i64 %indvars.iv449.epil to i32
  %mul35336.epil = mul i32 %indvars451.epil, %29
  %add37338.epil = add i32 %mul35336.epil, 3
  %add39.epil = mul i32 %add37338.epil, %29
  %arrayidx41.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv449.epil
  store i32 %add39.epil, ptr %arrayidx41.epil, align 4, !tbaa !4
  %epil.iter530.next = add i64 %epil.iter530, 1
  %epil.iter530.cmp.not = icmp eq i64 %epil.iter530.next, %xtraiter529
  br i1 %epil.iter530.cmp.not, label %sw.epilog, label %for.body33.epil, !llvm.loop !18

sw.epilog.loopexit481.unr-lcssa:                  ; preds = %for.body52, %for.body52.preheader
  %indvars.iv442.unr = phi i64 [ 0, %for.body52.preheader ], [ %indvars.iv.next443.3, %for.body52 ]
  %lcmp.mod526.not = icmp eq i64 %xtraiter524, 0
  br i1 %lcmp.mod526.not, label %sw.epilog, label %for.body52.epil

for.body52.epil:                                  ; preds = %sw.epilog.loopexit481.unr-lcssa, %for.body52.epil
  %indvars.iv442.epil = phi i64 [ %indvars.iv.next443.epil, %for.body52.epil ], [ %indvars.iv442.unr, %sw.epilog.loopexit481.unr-lcssa ]
  %epil.iter525 = phi i64 [ %epil.iter525.next, %for.body52.epil ], [ 0, %sw.epilog.loopexit481.unr-lcssa ]
  %indvars446.epil = trunc i64 %indvars.iv442.epil to i32
  %i47.0333.epil = add nsw i32 %indvars446.epil, -1
  %mul54332.epil = mul i32 %i47.0333.epil, %indvars446.epil
  %sub334.epil = add i32 %mul54332.epil, 3
  %add57.epil = mul i32 %sub334.epil, %indvars446.epil
  %arrayidx59.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv442.epil
  store i32 %add57.epil, ptr %arrayidx59.epil, align 4, !tbaa !4
  %indvars.iv.next443.epil = add nuw nsw i64 %indvars.iv442.epil, 1
  %epil.iter525.next = add i64 %epil.iter525, 1
  %epil.iter525.cmp.not = icmp eq i64 %epil.iter525.next, %xtraiter524
  br i1 %epil.iter525.cmp.not, label %sw.epilog, label %for.body52.epil, !llvm.loop !19

sw.epilog.loopexit482.unr-lcssa:                  ; preds = %for.body70, %for.body70.preheader
  %indvars.iv436.unr = phi i64 [ 0, %for.body70.preheader ], [ %indvars.iv.next437.3, %for.body70 ]
  %lcmp.mod521.not = icmp eq i64 %xtraiter519, 0
  br i1 %lcmp.mod521.not, label %sw.epilog, label %for.body70.epil

for.body70.epil:                                  ; preds = %sw.epilog.loopexit482.unr-lcssa, %for.body70.epil
  %indvars.iv436.epil = phi i64 [ %indvars.iv.next437.epil, %for.body70.epil ], [ %indvars.iv436.unr, %sw.epilog.loopexit482.unr-lcssa ]
  %epil.iter520 = phi i64 [ %epil.iter520.next, %for.body70.epil ], [ 0, %sw.epilog.loopexit482.unr-lcssa ]
  %indvars.iv.next437.epil = add nuw nsw i64 %indvars.iv436.epil, 1
  %indvars438.epil = trunc i64 %indvars.iv.next437.epil to i32
  %30 = trunc nuw nsw i64 %indvars.iv436.epil to i32
  %add72.epil = mul i32 %indvars438.epil, %30
  %add73.epil = add nsw i32 %add72.epil, 3
  %arrayidx75.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv436.epil
  store i32 %add73.epil, ptr %arrayidx75.epil, align 4, !tbaa !4
  %epil.iter520.next = add i64 %epil.iter520, 1
  %epil.iter520.cmp.not = icmp eq i64 %epil.iter520.next, %xtraiter519
  br i1 %epil.iter520.cmp.not, label %sw.epilog, label %for.body70.epil, !llvm.loop !20

sw.epilog.loopexit483.unr-lcssa:                  ; preds = %for.body86, %for.body86.preheader
  %indvars.iv428.unr = phi i64 [ 0, %for.body86.preheader ], [ %indvars.iv.next429.3, %for.body86 ]
  %lcmp.mod516.not = icmp eq i64 %xtraiter514, 0
  br i1 %lcmp.mod516.not, label %sw.epilog, label %for.body86.epil

for.body86.epil:                                  ; preds = %sw.epilog.loopexit483.unr-lcssa, %for.body86.epil
  %indvars.iv428.epil = phi i64 [ %indvars.iv.next429.epil, %for.body86.epil ], [ %indvars.iv428.unr, %sw.epilog.loopexit483.unr-lcssa ]
  %epil.iter515 = phi i64 [ %epil.iter515.next, %for.body86.epil ], [ 0, %sw.epilog.loopexit483.unr-lcssa ]
  %indvars433.epil = trunc i64 %indvars.iv428.epil to i32
  %mul87.epil = mul nuw nsw i32 %indvars433.epil, %indvars433.epil
  %mul88327.epil = add nuw i32 %mul87.epil, 1
  %mul89326.epil = mul i32 %mul88327.epil, %indvars433.epil
  %add91328.epil = add i32 %mul89326.epil, 3
  %add93.epil = mul i32 %add91328.epil, %indvars433.epil
  %arrayidx95.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv428.epil
  store i32 %add93.epil, ptr %arrayidx95.epil, align 4, !tbaa !4
  %indvars.iv.next429.epil = add nuw nsw i64 %indvars.iv428.epil, 1
  %epil.iter515.next = add i64 %epil.iter515, 1
  %epil.iter515.cmp.not = icmp eq i64 %epil.iter515.next, %xtraiter514
  br i1 %epil.iter515.cmp.not, label %sw.epilog, label %for.body86.epil, !llvm.loop !21

sw.epilog:                                        ; preds = %sw.epilog.loopexit483.unr-lcssa, %for.body86.epil, %sw.epilog.loopexit482.unr-lcssa, %for.body70.epil, %sw.epilog.loopexit481.unr-lcssa, %for.body52.epil, %sw.epilog.loopexit480.unr-lcssa, %for.body33.epil, %sw.epilog.loopexit.unr-lcssa, %for.body17.epil, %sw.bb79, %sw.bb63, %sw.bb45, %sw.bb26, %sw.bb, %if.then9
  call void @func4()
  unreachable

if.else:                                          ; preds = %if.then
  %puts308 = call i32 @puts(ptr nonnull dereferenceable(1) @str.21)
  %31 = load i32, ptr %len, align 4, !tbaa !4
  %cmp102354 = icmp sgt i32 %31, 0
  br i1 %cmp102354, label %for.body105, label %for.cond.cleanup104

for.cond.cleanup104:                              ; preds = %for.body105, %if.else
  %.lcssa = phi i32 [ %31, %if.else ], [ %33, %for.body105 ]
  %rem112 = srem i32 %.lcssa, 5
  switch i32 %rem112, label %sw.epilog201 [
    i32 0, label %sw.bb113
    i32 1, label %sw.bb133
    i32 2, label %sw.bb149
    i32 3, label %sw.bb166
    i32 4, label %sw.bb185
  ]

for.body105:                                      ; preds = %if.else, %for.body105
  %indvars.iv390 = phi i64 [ %indvars.iv.next391, %for.body105 ], [ 0, %if.else ]
  %arrayidx107 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv390
  %32 = load i32, ptr %arrayidx107, align 4, !tbaa !4
  %call108 = call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.13, i32 noundef signext %32)
  %indvars.iv.next391 = add nuw nsw i64 %indvars.iv390, 1
  %33 = load i32, ptr %len, align 4, !tbaa !4
  %34 = sext i32 %33 to i64
  %cmp102 = icmp slt i64 %indvars.iv.next391, %34
  br i1 %cmp102, label %for.body105, label %for.cond.cleanup104, !llvm.loop !22

sw.bb113:                                         ; preds = %for.cond.cleanup104
  %puts320 = call i32 @puts(ptr nonnull dereferenceable(1) @str.32)
  %35 = load i32, ptr %len, align 4, !tbaa !4
  %cmp117364 = icmp sgt i32 %35, 0
  br i1 %cmp117364, label %for.body120.preheader, label %sw.epilog201

for.body120.preheader:                            ; preds = %sw.bb113
  %wide.trip.count426 = zext nneg i32 %35 to i64
  %xtraiter509 = and i64 %wide.trip.count426, 3
  %36 = icmp ult i32 %35, 4
  br i1 %36, label %sw.epilog201.loopexit.unr-lcssa, label %for.body120.preheader.new

for.body120.preheader.new:                        ; preds = %for.body120.preheader
  %unroll_iter512 = and i64 %wide.trip.count426, 2147483644
  br label %for.body120

for.body120:                                      ; preds = %for.body120, %for.body120.preheader.new
  %indvars.iv420 = phi i64 [ 0, %for.body120.preheader.new ], [ %indvars.iv.next421.3, %for.body120 ]
  %niter513 = phi i64 [ 0, %for.body120.preheader.new ], [ %niter513.next.3, %for.body120 ]
  %indvars425 = trunc i64 %indvars.iv420 to i32
  %mul121 = mul nuw nsw i32 %indvars425, %indvars425
  %mul122322 = or disjoint i32 %mul121, 1
  %mul123321 = mul i32 %mul122322, %indvars425
  %add125323 = or disjoint i32 %mul123321, 3
  %add127 = mul i32 %add125323, %indvars425
  %arrayidx129 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv420
  store i32 %add127, ptr %arrayidx129, align 8, !tbaa !4
  %indvars.iv.next421 = or disjoint i64 %indvars.iv420, 1
  %indvars425.1 = trunc i64 %indvars.iv.next421 to i32
  %mul121.1 = mul nuw nsw i32 %indvars425.1, %indvars425.1
  %mul122322.1 = add nuw nsw i32 %mul121.1, 1
  %mul123321.1 = mul i32 %mul122322.1, %indvars425.1
  %add125323.1 = add i32 %mul123321.1, 3
  %add127.1 = mul i32 %add125323.1, %indvars425.1
  %arrayidx129.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next421
  store i32 %add127.1, ptr %arrayidx129.1, align 4, !tbaa !4
  %indvars.iv.next421.1 = or disjoint i64 %indvars.iv420, 2
  %indvars425.2 = trunc i64 %indvars.iv.next421.1 to i32
  %mul121.2 = mul nuw nsw i32 %indvars425.2, %indvars425.2
  %mul122322.2 = or disjoint i32 %mul121.2, 1
  %mul123321.2 = mul i32 %mul122322.2, %indvars425.2
  %add125323.2 = add i32 %mul123321.2, 3
  %add127.2 = mul i32 %add125323.2, %indvars425.2
  %arrayidx129.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next421.1
  store i32 %add127.2, ptr %arrayidx129.2, align 8, !tbaa !4
  %indvars.iv.next421.2 = or disjoint i64 %indvars.iv420, 3
  %indvars425.3 = trunc i64 %indvars.iv.next421.2 to i32
  %mul121.3 = mul nuw nsw i32 %indvars425.3, %indvars425.3
  %mul122322.3 = add nuw nsw i32 %mul121.3, 1
  %mul123321.3 = mul i32 %mul122322.3, %indvars425.3
  %add125323.3 = add i32 %mul123321.3, 3
  %add127.3 = mul i32 %add125323.3, %indvars425.3
  %arrayidx129.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next421.2
  store i32 %add127.3, ptr %arrayidx129.3, align 4, !tbaa !4
  %indvars.iv.next421.3 = add nuw nsw i64 %indvars.iv420, 4
  %niter513.next.3 = add i64 %niter513, 4
  %niter513.ncmp.3 = icmp eq i64 %niter513.next.3, %unroll_iter512
  br i1 %niter513.ncmp.3, label %sw.epilog201.loopexit.unr-lcssa, label %for.body120, !llvm.loop !23

sw.bb133:                                         ; preds = %for.cond.cleanup104
  %puts318 = call i32 @puts(ptr nonnull dereferenceable(1) @str.31)
  %37 = load i32, ptr %len, align 4, !tbaa !4
  %cmp137362 = icmp sgt i32 %37, 0
  br i1 %cmp137362, label %for.body140.preheader, label %sw.epilog201

for.body140.preheader:                            ; preds = %sw.bb133
  %wide.trip.count418 = zext nneg i32 %37 to i64
  %xtraiter504 = and i64 %wide.trip.count418, 3
  %38 = icmp ult i32 %37, 4
  br i1 %38, label %sw.epilog201.loopexit484.unr-lcssa, label %for.body140.preheader.new

for.body140.preheader.new:                        ; preds = %for.body140.preheader
  %unroll_iter507 = and i64 %wide.trip.count418, 2147483644
  br label %for.body140

for.body140:                                      ; preds = %for.body140, %for.body140.preheader.new
  %indvars.iv414 = phi i64 [ 0, %for.body140.preheader.new ], [ %indvars.iv.next415.3, %for.body140 ]
  %niter508 = phi i64 [ 0, %for.body140.preheader.new ], [ %niter508.next.3, %for.body140 ]
  %indvars.iv.next415 = or disjoint i64 %indvars.iv414, 1
  %indvars416 = trunc i64 %indvars.iv.next415 to i32
  %39 = trunc nuw nsw i64 %indvars.iv414 to i32
  %add142 = mul i32 %indvars416, %39
  %add143 = or disjoint i32 %add142, 3
  %arrayidx145 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv414
  store i32 %add143, ptr %arrayidx145, align 8, !tbaa !4
  %indvars.iv.next415.1 = or disjoint i64 %indvars.iv414, 2
  %indvars416.1 = trunc i64 %indvars.iv.next415.1 to i32
  %40 = trunc nuw nsw i64 %indvars.iv.next415 to i32
  %add142.1 = mul i32 %indvars416.1, %40
  %add143.1 = add nsw i32 %add142.1, 3
  %arrayidx145.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next415
  store i32 %add143.1, ptr %arrayidx145.1, align 4, !tbaa !4
  %indvars.iv.next415.2 = or disjoint i64 %indvars.iv414, 3
  %indvars416.2 = trunc i64 %indvars.iv.next415.2 to i32
  %41 = trunc nuw nsw i64 %indvars.iv.next415.1 to i32
  %add142.2 = mul i32 %indvars416.2, %41
  %add143.2 = add nsw i32 %add142.2, 3
  %arrayidx145.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next415.1
  store i32 %add143.2, ptr %arrayidx145.2, align 8, !tbaa !4
  %indvars.iv.next415.3 = add nuw nsw i64 %indvars.iv414, 4
  %indvars416.3 = trunc i64 %indvars.iv.next415.3 to i32
  %42 = trunc nuw nsw i64 %indvars.iv.next415.2 to i32
  %add142.3 = mul i32 %indvars416.3, %42
  %add143.3 = or disjoint i32 %add142.3, 3
  %arrayidx145.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next415.2
  store i32 %add143.3, ptr %arrayidx145.3, align 4, !tbaa !4
  %niter508.next.3 = add i64 %niter508, 4
  %niter508.ncmp.3 = icmp eq i64 %niter508.next.3, %unroll_iter507
  br i1 %niter508.ncmp.3, label %sw.epilog201.loopexit484.unr-lcssa, label %for.body140, !llvm.loop !24

sw.bb149:                                         ; preds = %for.cond.cleanup104
  %puts315 = call i32 @puts(ptr nonnull dereferenceable(1) @str.30)
  %43 = load i32, ptr %len, align 4, !tbaa !4
  %cmp153360 = icmp sgt i32 %43, 0
  br i1 %cmp153360, label %for.body156.preheader, label %sw.epilog201

for.body156.preheader:                            ; preds = %sw.bb149
  %wide.trip.count412 = zext nneg i32 %43 to i64
  %xtraiter499 = and i64 %wide.trip.count412, 3
  %44 = icmp ult i32 %43, 4
  br i1 %44, label %sw.epilog201.loopexit485.unr-lcssa, label %for.body156.preheader.new

for.body156.preheader.new:                        ; preds = %for.body156.preheader
  %unroll_iter502 = and i64 %wide.trip.count412, 2147483644
  br label %for.body156

for.body156:                                      ; preds = %for.body156, %for.body156.preheader.new
  %indvars.iv407 = phi i64 [ 0, %for.body156.preheader.new ], [ %indvars.iv.next408.3, %for.body156 ]
  %niter503 = phi i64 [ 0, %for.body156.preheader.new ], [ %niter503.next.3, %for.body156 ]
  %45 = trunc nuw nsw i64 %indvars.iv407 to i32
  %mul158316 = mul i32 %45, %45
  %46 = trunc i64 %indvars.iv407 to i32
  %47 = add i32 %46, -1
  %sub160 = mul i32 %mul158316, %47
  %arrayidx162 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv407
  store i32 %sub160, ptr %arrayidx162, align 8, !tbaa !4
  %indvars.iv.next408 = or disjoint i64 %indvars.iv407, 1
  %48 = trunc nuw nsw i64 %indvars.iv.next408 to i32
  %mul158316.1 = mul i32 %48, %48
  %49 = trunc i64 %indvars.iv.next408 to i32
  %50 = add nsw i32 %49, -1
  %sub160.1 = mul i32 %mul158316.1, %50
  %arrayidx162.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next408
  store i32 %sub160.1, ptr %arrayidx162.1, align 4, !tbaa !4
  %indvars.iv.next408.1 = or disjoint i64 %indvars.iv407, 2
  %51 = trunc nuw nsw i64 %indvars.iv.next408.1 to i32
  %mul158316.2 = mul i32 %51, %51
  %52 = trunc i64 %indvars.iv.next408.1 to i32
  %53 = add nsw i32 %52, -1
  %sub160.2 = mul i32 %mul158316.2, %53
  %arrayidx162.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next408.1
  store i32 %sub160.2, ptr %arrayidx162.2, align 8, !tbaa !4
  %indvars.iv.next408.2 = or disjoint i64 %indvars.iv407, 3
  %54 = trunc nuw nsw i64 %indvars.iv.next408.2 to i32
  %mul158316.3 = mul i32 %54, %54
  %55 = trunc i64 %indvars.iv.next408.2 to i32
  %56 = add nsw i32 %55, -1
  %sub160.3 = mul i32 %mul158316.3, %56
  %arrayidx162.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next408.2
  store i32 %sub160.3, ptr %arrayidx162.3, align 4, !tbaa !4
  %indvars.iv.next408.3 = add nuw nsw i64 %indvars.iv407, 4
  %niter503.next.3 = add i64 %niter503, 4
  %niter503.ncmp.3 = icmp eq i64 %niter503.next.3, %unroll_iter502
  br i1 %niter503.ncmp.3, label %sw.epilog201.loopexit485.unr-lcssa, label %for.body156, !llvm.loop !25

sw.bb166:                                         ; preds = %for.cond.cleanup104
  %puts311 = call i32 @puts(ptr nonnull dereferenceable(1) @str.29)
  %57 = load i32, ptr %len, align 4, !tbaa !4
  %cmp170358 = icmp sgt i32 %57, 0
  br i1 %cmp170358, label %for.body173.preheader, label %sw.epilog201

for.body173.preheader:                            ; preds = %sw.bb166
  %wide.trip.count405 = zext nneg i32 %57 to i64
  %xtraiter494 = and i64 %wide.trip.count405, 3
  %58 = icmp ult i32 %57, 4
  br i1 %58, label %sw.epilog201.loopexit486.unr-lcssa, label %for.body173.preheader.new

for.body173.preheader.new:                        ; preds = %for.body173.preheader
  %unroll_iter497 = and i64 %wide.trip.count405, 2147483644
  br label %for.body173

for.body173:                                      ; preds = %for.body173, %for.body173.preheader.new
  %indvars.iv401 = phi i64 [ 0, %for.body173.preheader.new ], [ %indvars.iv.next402.3, %for.body173 ]
  %niter498 = phi i64 [ 0, %for.body173.preheader.new ], [ %niter498.next.3, %for.body173 ]
  %indvars.iv.next402 = or disjoint i64 %indvars.iv401, 1
  %indvars403 = trunc i64 %indvars.iv.next402 to i32
  %59 = trunc nuw nsw i64 %indvars.iv401 to i32
  %mul175312 = mul i32 %indvars403, %59
  %add177314 = or disjoint i32 %mul175312, 3
  %add179 = mul i32 %add177314, %59
  %arrayidx181 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv401
  store i32 %add179, ptr %arrayidx181, align 8, !tbaa !4
  %indvars.iv.next402.1 = or disjoint i64 %indvars.iv401, 2
  %indvars403.1 = trunc i64 %indvars.iv.next402.1 to i32
  %60 = trunc nuw nsw i64 %indvars.iv.next402 to i32
  %mul175312.1 = mul i32 %indvars403.1, %60
  %add177314.1 = add i32 %mul175312.1, 3
  %add179.1 = mul i32 %add177314.1, %60
  %arrayidx181.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next402
  store i32 %add179.1, ptr %arrayidx181.1, align 4, !tbaa !4
  %indvars.iv.next402.2 = or disjoint i64 %indvars.iv401, 3
  %indvars403.2 = trunc i64 %indvars.iv.next402.2 to i32
  %61 = trunc nuw nsw i64 %indvars.iv.next402.1 to i32
  %mul175312.2 = mul i32 %indvars403.2, %61
  %add177314.2 = add i32 %mul175312.2, 3
  %add179.2 = mul i32 %add177314.2, %61
  %arrayidx181.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next402.1
  store i32 %add179.2, ptr %arrayidx181.2, align 8, !tbaa !4
  %indvars.iv.next402.3 = add nuw nsw i64 %indvars.iv401, 4
  %indvars403.3 = trunc i64 %indvars.iv.next402.3 to i32
  %62 = trunc nuw nsw i64 %indvars.iv.next402.2 to i32
  %mul175312.3 = mul i32 %indvars403.3, %62
  %add177314.3 = or disjoint i32 %mul175312.3, 3
  %add179.3 = mul i32 %add177314.3, %62
  %arrayidx181.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next402.2
  store i32 %add179.3, ptr %arrayidx181.3, align 4, !tbaa !4
  %niter498.next.3 = add i64 %niter498, 4
  %niter498.ncmp.3 = icmp eq i64 %niter498.next.3, %unroll_iter497
  br i1 %niter498.ncmp.3, label %sw.epilog201.loopexit486.unr-lcssa, label %for.body173, !llvm.loop !26

sw.bb185:                                         ; preds = %for.cond.cleanup104
  %puts309 = call i32 @puts(ptr nonnull dereferenceable(1) @str.28)
  %63 = load i32, ptr %len, align 4, !tbaa !4
  %cmp189356 = icmp sgt i32 %63, 0
  br i1 %cmp189356, label %for.body192.preheader, label %sw.epilog201

for.body192.preheader:                            ; preds = %sw.bb185
  %wide.trip.count399 = zext nneg i32 %63 to i64
  %xtraiter489 = and i64 %wide.trip.count399, 3
  %64 = icmp ult i32 %63, 4
  br i1 %64, label %sw.epilog201.loopexit487.unr-lcssa, label %for.body192.preheader.new

for.body192.preheader.new:                        ; preds = %for.body192.preheader
  %unroll_iter492 = and i64 %wide.trip.count399, 2147483644
  br label %for.body192

for.body192:                                      ; preds = %for.body192, %for.body192.preheader.new
  %indvars.iv394 = phi i64 [ 0, %for.body192.preheader.new ], [ %indvars.iv.next395.3, %for.body192 ]
  %niter493 = phi i64 [ 0, %for.body192.preheader.new ], [ %niter493.next.3, %for.body192 ]
  %indvars398 = trunc i64 %indvars.iv394 to i32
  %mul194310 = or disjoint i32 %indvars398, 3
  %add195 = mul i32 %mul194310, %indvars398
  %arrayidx197 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv394
  store i32 %add195, ptr %arrayidx197, align 8, !tbaa !4
  %indvars.iv.next395 = or disjoint i64 %indvars.iv394, 1
  %indvars398.1 = trunc i64 %indvars.iv.next395 to i32
  %mul194310.1 = add nuw i32 %indvars398.1, 3
  %add195.1 = mul i32 %mul194310.1, %indvars398.1
  %arrayidx197.1 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next395
  store i32 %add195.1, ptr %arrayidx197.1, align 4, !tbaa !4
  %indvars.iv.next395.1 = or disjoint i64 %indvars.iv394, 2
  %indvars398.2 = trunc i64 %indvars.iv.next395.1 to i32
  %mul194310.2 = add nuw i32 %indvars398.2, 3
  %add195.2 = mul i32 %mul194310.2, %indvars398.2
  %arrayidx197.2 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next395.1
  store i32 %add195.2, ptr %arrayidx197.2, align 8, !tbaa !4
  %indvars.iv.next395.2 = or disjoint i64 %indvars.iv394, 3
  %indvars398.3 = trunc i64 %indvars.iv.next395.2 to i32
  %mul194310.3 = add nuw i32 %indvars398.3, 3
  %add195.3 = mul i32 %mul194310.3, %indvars398.3
  %arrayidx197.3 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv.next395.2
  store i32 %add195.3, ptr %arrayidx197.3, align 4, !tbaa !4
  %indvars.iv.next395.3 = add nuw nsw i64 %indvars.iv394, 4
  %niter493.next.3 = add i64 %niter493, 4
  %niter493.ncmp.3 = icmp eq i64 %niter493.next.3, %unroll_iter492
  br i1 %niter493.ncmp.3, label %sw.epilog201.loopexit487.unr-lcssa, label %for.body192, !llvm.loop !27

sw.epilog201.loopexit.unr-lcssa:                  ; preds = %for.body120, %for.body120.preheader
  %indvars.iv420.unr = phi i64 [ 0, %for.body120.preheader ], [ %indvars.iv.next421.3, %for.body120 ]
  %lcmp.mod511.not = icmp eq i64 %xtraiter509, 0
  br i1 %lcmp.mod511.not, label %sw.epilog201, label %for.body120.epil

for.body120.epil:                                 ; preds = %sw.epilog201.loopexit.unr-lcssa, %for.body120.epil
  %indvars.iv420.epil = phi i64 [ %indvars.iv.next421.epil, %for.body120.epil ], [ %indvars.iv420.unr, %sw.epilog201.loopexit.unr-lcssa ]
  %epil.iter510 = phi i64 [ %epil.iter510.next, %for.body120.epil ], [ 0, %sw.epilog201.loopexit.unr-lcssa ]
  %indvars425.epil = trunc i64 %indvars.iv420.epil to i32
  %mul121.epil = mul nuw nsw i32 %indvars425.epil, %indvars425.epil
  %mul122322.epil = add nuw i32 %mul121.epil, 1
  %mul123321.epil = mul i32 %mul122322.epil, %indvars425.epil
  %add125323.epil = add i32 %mul123321.epil, 3
  %add127.epil = mul i32 %add125323.epil, %indvars425.epil
  %arrayidx129.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv420.epil
  store i32 %add127.epil, ptr %arrayidx129.epil, align 4, !tbaa !4
  %indvars.iv.next421.epil = add nuw nsw i64 %indvars.iv420.epil, 1
  %epil.iter510.next = add i64 %epil.iter510, 1
  %epil.iter510.cmp.not = icmp eq i64 %epil.iter510.next, %xtraiter509
  br i1 %epil.iter510.cmp.not, label %sw.epilog201, label %for.body120.epil, !llvm.loop !28

sw.epilog201.loopexit484.unr-lcssa:               ; preds = %for.body140, %for.body140.preheader
  %indvars.iv414.unr = phi i64 [ 0, %for.body140.preheader ], [ %indvars.iv.next415.3, %for.body140 ]
  %lcmp.mod506.not = icmp eq i64 %xtraiter504, 0
  br i1 %lcmp.mod506.not, label %sw.epilog201, label %for.body140.epil

for.body140.epil:                                 ; preds = %sw.epilog201.loopexit484.unr-lcssa, %for.body140.epil
  %indvars.iv414.epil = phi i64 [ %indvars.iv.next415.epil, %for.body140.epil ], [ %indvars.iv414.unr, %sw.epilog201.loopexit484.unr-lcssa ]
  %epil.iter505 = phi i64 [ %epil.iter505.next, %for.body140.epil ], [ 0, %sw.epilog201.loopexit484.unr-lcssa ]
  %indvars.iv.next415.epil = add nuw nsw i64 %indvars.iv414.epil, 1
  %indvars416.epil = trunc i64 %indvars.iv.next415.epil to i32
  %65 = trunc nuw nsw i64 %indvars.iv414.epil to i32
  %add142.epil = mul i32 %indvars416.epil, %65
  %add143.epil = add nsw i32 %add142.epil, 3
  %arrayidx145.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv414.epil
  store i32 %add143.epil, ptr %arrayidx145.epil, align 4, !tbaa !4
  %epil.iter505.next = add i64 %epil.iter505, 1
  %epil.iter505.cmp.not = icmp eq i64 %epil.iter505.next, %xtraiter504
  br i1 %epil.iter505.cmp.not, label %sw.epilog201, label %for.body140.epil, !llvm.loop !29

sw.epilog201.loopexit485.unr-lcssa:               ; preds = %for.body156, %for.body156.preheader
  %indvars.iv407.unr = phi i64 [ 0, %for.body156.preheader ], [ %indvars.iv.next408.3, %for.body156 ]
  %lcmp.mod501.not = icmp eq i64 %xtraiter499, 0
  br i1 %lcmp.mod501.not, label %sw.epilog201, label %for.body156.epil

for.body156.epil:                                 ; preds = %sw.epilog201.loopexit485.unr-lcssa, %for.body156.epil
  %indvars.iv407.epil = phi i64 [ %indvars.iv.next408.epil, %for.body156.epil ], [ %indvars.iv407.unr, %sw.epilog201.loopexit485.unr-lcssa ]
  %epil.iter500 = phi i64 [ %epil.iter500.next, %for.body156.epil ], [ 0, %sw.epilog201.loopexit485.unr-lcssa ]
  %66 = trunc nuw nsw i64 %indvars.iv407.epil to i32
  %mul158316.epil = mul i32 %66, %66
  %67 = trunc i64 %indvars.iv407.epil to i32
  %68 = add i32 %67, -1
  %sub160.epil = mul i32 %mul158316.epil, %68
  %arrayidx162.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv407.epil
  store i32 %sub160.epil, ptr %arrayidx162.epil, align 4, !tbaa !4
  %indvars.iv.next408.epil = add nuw nsw i64 %indvars.iv407.epil, 1
  %epil.iter500.next = add i64 %epil.iter500, 1
  %epil.iter500.cmp.not = icmp eq i64 %epil.iter500.next, %xtraiter499
  br i1 %epil.iter500.cmp.not, label %sw.epilog201, label %for.body156.epil, !llvm.loop !30

sw.epilog201.loopexit486.unr-lcssa:               ; preds = %for.body173, %for.body173.preheader
  %indvars.iv401.unr = phi i64 [ 0, %for.body173.preheader ], [ %indvars.iv.next402.3, %for.body173 ]
  %lcmp.mod496.not = icmp eq i64 %xtraiter494, 0
  br i1 %lcmp.mod496.not, label %sw.epilog201, label %for.body173.epil

for.body173.epil:                                 ; preds = %sw.epilog201.loopexit486.unr-lcssa, %for.body173.epil
  %indvars.iv401.epil = phi i64 [ %indvars.iv.next402.epil, %for.body173.epil ], [ %indvars.iv401.unr, %sw.epilog201.loopexit486.unr-lcssa ]
  %epil.iter495 = phi i64 [ %epil.iter495.next, %for.body173.epil ], [ 0, %sw.epilog201.loopexit486.unr-lcssa ]
  %indvars.iv.next402.epil = add nuw nsw i64 %indvars.iv401.epil, 1
  %indvars403.epil = trunc i64 %indvars.iv.next402.epil to i32
  %69 = trunc nuw nsw i64 %indvars.iv401.epil to i32
  %mul175312.epil = mul i32 %indvars403.epil, %69
  %add177314.epil = add i32 %mul175312.epil, 3
  %add179.epil = mul i32 %add177314.epil, %69
  %arrayidx181.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv401.epil
  store i32 %add179.epil, ptr %arrayidx181.epil, align 4, !tbaa !4
  %epil.iter495.next = add i64 %epil.iter495, 1
  %epil.iter495.cmp.not = icmp eq i64 %epil.iter495.next, %xtraiter494
  br i1 %epil.iter495.cmp.not, label %sw.epilog201, label %for.body173.epil, !llvm.loop !31

sw.epilog201.loopexit487.unr-lcssa:               ; preds = %for.body192, %for.body192.preheader
  %indvars.iv394.unr = phi i64 [ 0, %for.body192.preheader ], [ %indvars.iv.next395.3, %for.body192 ]
  %lcmp.mod491.not = icmp eq i64 %xtraiter489, 0
  br i1 %lcmp.mod491.not, label %sw.epilog201, label %for.body192.epil

for.body192.epil:                                 ; preds = %sw.epilog201.loopexit487.unr-lcssa, %for.body192.epil
  %indvars.iv394.epil = phi i64 [ %indvars.iv.next395.epil, %for.body192.epil ], [ %indvars.iv394.unr, %sw.epilog201.loopexit487.unr-lcssa ]
  %epil.iter490 = phi i64 [ %epil.iter490.next, %for.body192.epil ], [ 0, %sw.epilog201.loopexit487.unr-lcssa ]
  %indvars398.epil = trunc i64 %indvars.iv394.epil to i32
  %mul194310.epil = add nuw i32 %indvars398.epil, 3
  %add195.epil = mul i32 %mul194310.epil, %indvars398.epil
  %arrayidx197.epil = getelementptr inbounds i32, ptr %1, i64 %indvars.iv394.epil
  store i32 %add195.epil, ptr %arrayidx197.epil, align 4, !tbaa !4
  %indvars.iv.next395.epil = add nuw nsw i64 %indvars.iv394.epil, 1
  %epil.iter490.next = add i64 %epil.iter490, 1
  %epil.iter490.cmp.not = icmp eq i64 %epil.iter490.next, %xtraiter489
  br i1 %epil.iter490.cmp.not, label %sw.epilog201, label %for.body192.epil, !llvm.loop !32

sw.epilog201:                                     ; preds = %sw.epilog201.loopexit487.unr-lcssa, %for.body192.epil, %sw.epilog201.loopexit486.unr-lcssa, %for.body173.epil, %sw.epilog201.loopexit485.unr-lcssa, %for.body156.epil, %sw.epilog201.loopexit484.unr-lcssa, %for.body140.epil, %sw.epilog201.loopexit.unr-lcssa, %for.body120.epil, %sw.bb185, %sw.bb166, %sw.bb149, %sw.bb133, %sw.bb113, %for.cond.cleanup104
  call void @func3()
  unreachable

if.else202:                                       ; preds = %for.cond.cleanup
  %puts = call i32 @puts(ptr nonnull dereferenceable(1) @str.19)
  %70 = load i32, ptr %len, align 4, !tbaa !4
  %cmp206352 = icmp sgt i32 %70, 0
  br i1 %cmp206352, label %for.body209, label %for.cond.cleanup208

for.cond.cleanup208:                              ; preds = %for.body209, %if.else202
  call void @func2()
  unreachable

for.body209:                                      ; preds = %if.else202, %for.body209
  %indvars.iv386 = phi i64 [ %indvars.iv.next387, %for.body209 ], [ 0, %if.else202 ]
  %arrayidx211 = getelementptr inbounds i32, ptr %1, i64 %indvars.iv386
  %71 = load i32, ptr %arrayidx211, align 4, !tbaa !4
  %call212 = call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.13, i32 noundef signext %71)
  %indvars.iv.next387 = add nuw nsw i64 %indvars.iv386, 1
  %72 = load i32, ptr %len, align 4, !tbaa !4
  %73 = sext i32 %72 to i64
  %cmp206 = icmp slt i64 %indvars.iv.next387, %73
  br i1 %cmp206, label %for.body209, label %for.cond.cleanup208, !llvm.loop !33
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
; CHECK:        larl    %r1, buf1
; CHECK:        lg      %r2, 8(%r1)
; CHECK:        lg      %r11, 0(%r1)
; CHECK:        lg      %r13, 32(%r1)
; CHECK:        lg      %r3, 16(%r1)
; CHECK:        stg     %r3, 0(%r15)
; CHECK:        lg      %r15, 24(%r1)
; CHECK:        br      %r2

  %0 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf1)
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %puts3 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.34)
  %call1 = tail call signext i32 @func1()
  unreachable

if.else:                                          ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.33)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr nocapture noundef readonly) local_unnamed_addr #7

attributes #0 = { noinline noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #1 = { nofree nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #2 = { noreturn nounwind }
attributes #3 = { noreturn nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nounwind }
attributes #6 = { nounwind "backchain" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="z10" }
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
