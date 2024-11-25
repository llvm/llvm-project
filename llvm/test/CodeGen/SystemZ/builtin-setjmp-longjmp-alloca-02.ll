; Test for Frame Pointer in first slot in jmp_buf.
; Test assembly for store/load to/from for nested setjmp for alloca for
; setjmp/longjmp respectively.

; Test for Frame Pointer in first slot in jmp_buf.
; Test assembly for nested setjmp for alloca.

; Frame Pointer in slot 1.
; Return address in slot 2.
; Stack Pointer in slot 4.

; RUN: llc -O2 < %s | FileCheck %s

@buf3 = dso_local global [10 x ptr] zeroinitializer, align 8
@buf2 = dso_local global [10 x ptr] zeroinitializer, align 8
@buf1 = dso_local global [10 x ptr] zeroinitializer, align 8
@len = dso_local local_unnamed_addr global i32 10, align 4
@.str.6 = private unnamed_addr constant [9 x i8] c"arr: %d\0A\00", align 2
@str = private unnamed_addr constant [9 x i8] c"In func4\00", align 1
@str.10 = private unnamed_addr constant [9 x i8] c"In func3\00", align 1
@str.11 = private unnamed_addr constant [9 x i8] c"In func2\00", align 1
@str.12 = private unnamed_addr constant [20 x i8] c"Returned from func3\00", align 1
@str.13 = private unnamed_addr constant [32 x i8] c"First __builtin_setjmp in func1\00", align 1
@str.14 = private unnamed_addr constant [20 x i8] c"Returned from func4\00", align 1
@str.15 = private unnamed_addr constant [33 x i8] c"Second __builtin_setjmp in func1\00", align 1
@str.16 = private unnamed_addr constant [44 x i8] c"In main, after __builtin_longjmp from func1\00", align 1
@str.17 = private unnamed_addr constant [20 x i8] c"In main, first time\00", align 1

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
; Load Frame Pointer from slot 1.
; Load return address from slot 2.
; Load stack pointer from slot 4.
; Load literal  pointer from slot 5.

; CHECK: larl    %r1, buf2
; CHECK: lg      %r2, 8(%r1)
; CHECK: lg      %r11, 0(%r1)
; CHECK: lg      %r13, 32(%r1)
; CHECK: lg      %r15, 24(%r1)
; CHECK: br      %r2
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.10)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf2)
  unreachable
}

; Function Attrs: noinline noreturn nounwind
define dso_local void @func2() local_unnamed_addr #0 {
entry:
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.11)
  tail call void @llvm.eh.sjlj.longjmp(ptr nonnull @buf1)
  unreachable
}

; Function Attrs: noreturn nounwind
define dso_local noundef signext i32 @func1(i32 noundef signext %len) local_unnamed_addr #3 {
entry:
  %conv = sext i32 %len to i64
  %mul = shl nsw i64 %conv, 2
  %0 = alloca i8, i64 %mul, align 8
  %cmp84 = icmp sgt i32 %len, 0
  br i1 %cmp84, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext nneg i32 %len to i64
  %xtraiter = and i64 %wide.trip.count, 3
  %1 = icmp ult i32 %len, 4
  br i1 %1, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body.preheader.new

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
  %2 = trunc nuw nsw i64 %indvars.iv.epil to i32
  %add.epil = mul i32 %indvars.epil, %2
  %arrayidx.epil = getelementptr inbounds i32, ptr %0, i64 %indvars.iv.epil
  store i32 %add.epil, ptr %arrayidx.epil, align 4, !tbaa !4
  %epil.iter.next = add i64 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %for.cond.cleanup, label %for.body.epil, !llvm.loop !8

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit.unr-lcssa, %for.body.epil, %entry
; Store Frame Pointer in slot 1.
; Store Return address in slot 2.
; Store Stack Pointer in slot 4

; CHECK:        larl    %r0, .LBB3_13
; CHECK:        stg     %r0, 8(%r1)
; CHECK:        stg     %r11, 0(%r1)
; CHECK:        stg     %r15, 24(%r1)

  %3 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf2)
  %cmp3 = icmp eq i32 %3, 0
  br i1 %cmp3, label %if.then, label %if.else38

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %indvars.iv = phi i64 [ 0, %for.body.preheader.new ], [ %indvars.iv.next.3, %for.body ]
  %niter = phi i64 [ 0, %for.body.preheader.new ], [ %niter.next.3, %for.body ]
  %indvars.iv.next = or disjoint i64 %indvars.iv, 1
  %indvars = trunc i64 %indvars.iv.next to i32
  %4 = trunc nuw nsw i64 %indvars.iv to i32
  %add = mul i32 %indvars, %4
  %arrayidx = getelementptr inbounds i32, ptr %0, i64 %indvars.iv
  store i32 %add, ptr %arrayidx, align 8, !tbaa !4
  %indvars.iv.next.1 = or disjoint i64 %indvars.iv, 2
  %indvars.1 = trunc i64 %indvars.iv.next.1 to i32
  %5 = trunc nuw nsw i64 %indvars.iv.next to i32
  %add.1 = mul i32 %indvars.1, %5
  %arrayidx.1 = getelementptr inbounds i32, ptr %0, i64 %indvars.iv.next
  store i32 %add.1, ptr %arrayidx.1, align 4, !tbaa !4
  %indvars.iv.next.2 = or disjoint i64 %indvars.iv, 3
  %indvars.2 = trunc i64 %indvars.iv.next.2 to i32
  %6 = trunc nuw nsw i64 %indvars.iv.next.1 to i32
  %add.2 = mul i32 %indvars.2, %6
  %arrayidx.2 = getelementptr inbounds i32, ptr %0, i64 %indvars.iv.next.1
  store i32 %add.2, ptr %arrayidx.2, align 8, !tbaa !4
  %indvars.iv.next.3 = add nuw nsw i64 %indvars.iv, 4
  %indvars.3 = trunc i64 %indvars.iv.next.3 to i32
  %7 = trunc nuw nsw i64 %indvars.iv.next.2 to i32
  %add.3 = mul i32 %indvars.3, %7
  %arrayidx.3 = getelementptr inbounds i32, ptr %0, i64 %indvars.iv.next.2
  store i32 %add.3, ptr %arrayidx.3, align 4, !tbaa !4
  %niter.next.3 = add i64 %niter, 4
  %niter.ncmp.3 = icmp eq i64 %niter.next.3, %unroll_iter
  br i1 %niter.ncmp.3, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body, !llvm.loop !10

if.then:                                          ; preds = %for.cond.cleanup
  %puts77 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.13)
  %8 = tail call i32 @llvm.eh.sjlj.setjmp(ptr nonnull @buf3)
  %cmp5 = icmp eq i32 %8, 0
  br i1 %cmp5, label %if.then7, label %if.else

if.then7:                                         ; preds = %if.then
  %puts82 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.15)
  tail call void @func4()
  unreachable

if.else:                                          ; preds = %if.then
  %puts78 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.14)
  br i1 %cmp84, label %for.body15.preheader, label %for.cond.cleanup26

for.body15.preheader:                             ; preds = %if.else
  %wide.trip.count103 = zext nneg i32 %len to i64
  br label %for.body15

for.body27.preheader:                             ; preds = %for.body15
  %xtraiter111 = and i64 %wide.trip.count103, 3
  %9 = icmp ult i32 %len, 4
  br i1 %9, label %for.cond.cleanup26.loopexit.unr-lcssa, label %for.body27.preheader.new

for.body27.preheader.new:                         ; preds = %for.body27.preheader
  %unroll_iter114 = and i64 %wide.trip.count103, 2147483644
  br label %for.body27

for.body15:                                       ; preds = %for.body15.preheader, %for.body15
  %indvars.iv99 = phi i64 [ 0, %for.body15.preheader ], [ %indvars.iv.next100, %for.body15 ]
  %arrayidx17 = getelementptr inbounds i32, ptr %0, i64 %indvars.iv99
  %10 = load i32, ptr %arrayidx17, align 4, !tbaa !4
  %call18 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %10)
  %indvars.iv.next100 = add nuw nsw i64 %indvars.iv99, 1
  %exitcond104.not = icmp eq i64 %indvars.iv.next100, %wide.trip.count103
  br i1 %exitcond104.not, label %for.body27.preheader, label %for.body15, !llvm.loop !12

for.cond.cleanup26.loopexit.unr-lcssa:            ; preds = %for.body27, %for.body27.preheader
  %indvars.iv105.unr = phi i64 [ 0, %for.body27.preheader ], [ %indvars.iv.next106.3, %for.body27 ]
  %lcmp.mod113.not = icmp eq i64 %xtraiter111, 0
  br i1 %lcmp.mod113.not, label %for.cond.cleanup26, label %for.body27.epil

for.body27.epil:                                  ; preds = %for.cond.cleanup26.loopexit.unr-lcssa, %for.body27.epil
  %indvars.iv105.epil = phi i64 [ %indvars.iv.next106.epil, %for.body27.epil ], [ %indvars.iv105.unr, %for.cond.cleanup26.loopexit.unr-lcssa ]
  %epil.iter112 = phi i64 [ %epil.iter112.next, %for.body27.epil ], [ 0, %for.cond.cleanup26.loopexit.unr-lcssa ]
  %indvars.iv.next106.epil = add nuw nsw i64 %indvars.iv105.epil, 1
  %indvars107.epil = trunc i64 %indvars.iv.next106.epil to i32
  %11 = trunc nuw nsw i64 %indvars.iv105.epil to i32
  %mul2979.epil = mul i32 %indvars107.epil, %11
  %add3181.epil = add nuw nsw i32 %mul2979.epil, 1
  %add32.epil = mul i32 %add3181.epil, %11
  %arrayidx34.epil = getelementptr inbounds i32, ptr %0, i64 %indvars.iv105.epil
  store i32 %add32.epil, ptr %arrayidx34.epil, align 4, !tbaa !4
  %epil.iter112.next = add i64 %epil.iter112, 1
  %epil.iter112.cmp.not = icmp eq i64 %epil.iter112.next, %xtraiter111
  br i1 %epil.iter112.cmp.not, label %for.cond.cleanup26, label %for.body27.epil, !llvm.loop !13

for.cond.cleanup26:                               ; preds = %for.cond.cleanup26.loopexit.unr-lcssa, %for.body27.epil, %if.else
  tail call void @func3()
  unreachable

for.body27:                                       ; preds = %for.body27, %for.body27.preheader.new
  %indvars.iv105 = phi i64 [ 0, %for.body27.preheader.new ], [ %indvars.iv.next106.3, %for.body27 ]
  %niter115 = phi i64 [ 0, %for.body27.preheader.new ], [ %niter115.next.3, %for.body27 ]
  %indvars.iv.next106 = or disjoint i64 %indvars.iv105, 1
  %indvars107 = trunc i64 %indvars.iv.next106 to i32
  %12 = trunc nuw nsw i64 %indvars.iv105 to i32
  %mul2979 = mul i32 %indvars107, %12
  %add3181 = or disjoint i32 %mul2979, 1
  %add32 = mul i32 %add3181, %12
  %arrayidx34 = getelementptr inbounds i32, ptr %0, i64 %indvars.iv105
  store i32 %add32, ptr %arrayidx34, align 8, !tbaa !4
  %indvars.iv.next106.1 = or disjoint i64 %indvars.iv105, 2
  %indvars107.1 = trunc i64 %indvars.iv.next106.1 to i32
  %13 = trunc nuw nsw i64 %indvars.iv.next106 to i32
  %mul2979.1 = mul i32 %indvars107.1, %13
  %add3181.1 = or disjoint i32 %mul2979.1, 1
  %add32.1 = mul i32 %add3181.1, %13
  %arrayidx34.1 = getelementptr inbounds i32, ptr %0, i64 %indvars.iv.next106
  store i32 %add32.1, ptr %arrayidx34.1, align 4, !tbaa !4
  %indvars.iv.next106.2 = or disjoint i64 %indvars.iv105, 3
  %indvars107.2 = trunc i64 %indvars.iv.next106.2 to i32
  %14 = trunc nuw nsw i64 %indvars.iv.next106.1 to i32
  %mul2979.2 = mul i32 %indvars107.2, %14
  %add3181.2 = or disjoint i32 %mul2979.2, 1
  %add32.2 = mul i32 %add3181.2, %14
  %arrayidx34.2 = getelementptr inbounds i32, ptr %0, i64 %indvars.iv.next106.1
  store i32 %add32.2, ptr %arrayidx34.2, align 8, !tbaa !4
  %indvars.iv.next106.3 = add nuw nsw i64 %indvars.iv105, 4
  %indvars107.3 = trunc i64 %indvars.iv.next106.3 to i32
  %15 = trunc nuw nsw i64 %indvars.iv.next106.2 to i32
  %mul2979.3 = mul i32 %indvars107.3, %15
  %add3181.3 = or disjoint i32 %mul2979.3, 1
  %add32.3 = mul i32 %add3181.3, %15
  %arrayidx34.3 = getelementptr inbounds i32, ptr %0, i64 %indvars.iv.next106.2
  store i32 %add32.3, ptr %arrayidx34.3, align 4, !tbaa !4
  %niter115.next.3 = add i64 %niter115, 4
  %niter115.ncmp.3 = icmp eq i64 %niter115.next.3, %unroll_iter114
  br i1 %niter115.ncmp.3, label %for.cond.cleanup26.loopexit.unr-lcssa, label %for.body27, !llvm.loop !14

if.else38:                                        ; preds = %for.cond.cleanup
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.12)
  br i1 %cmp84, label %for.body45.preheader, label %for.cond.cleanup44

for.body45.preheader:                             ; preds = %if.else38
  %wide.trip.count97 = zext nneg i32 %len to i64
  br label %for.body45

for.cond.cleanup44:                               ; preds = %for.body45, %if.else38
  tail call void @func2()
  unreachable

for.body45:                                       ; preds = %for.body45.preheader, %for.body45
  %indvars.iv93 = phi i64 [ 0, %for.body45.preheader ], [ %indvars.iv.next94, %for.body45 ]
  %arrayidx47 = getelementptr inbounds i32, ptr %0, i64 %indvars.iv93
  %16 = load i32, ptr %arrayidx47, align 4, !tbaa !4
  %call48 = tail call signext i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef signext %16)
  %indvars.iv.next94 = add nuw nsw i64 %indvars.iv93, 1
  %exitcond98.not = icmp eq i64 %indvars.iv.next94, %wide.trip.count97
  br i1 %exitcond98.not, label %for.cond.cleanup44, label %for.body45, !llvm.loop !15
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
  %puts3 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.17)
  %1 = load i32, ptr @len, align 4, !tbaa !4
  %call1 = tail call signext i32 @func1(i32 noundef signext %1)
  unreachable

if.else:                                          ; preds = %entry
  %puts = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.16)
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


!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}

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

