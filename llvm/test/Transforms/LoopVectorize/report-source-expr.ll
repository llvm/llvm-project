; RUN: opt -report-source-expr=true -passes='function(loop-vectorize,require<access-info>)' -disable-output -pass-remarks-analysis=loop-vectorize  < %s 2>&1 | FileCheck %s



; // Dependence::Backward
; // Loop does not get vectorized since it contains a backward
; // dependency between A[i] and A[i+3].
; void test_backward_dep(int n, int *A) {
;   for (int i = 1; i <= n - 3; i += 3) {
;     A[i] = A[i-1];
;     A[i+1] = A[i+3];
;   }
; }

; CHECK:remark: source.c:4:14: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop Dependence source: &A[i] Dependence destination: &A[(i + 3)]
; CHECK-NEXT: Backward loop carried data dependence. Memory location is the same as accessed at source.c:3:6

define dso_local void @test_backward_dep(i32 noundef %n, ptr nocapture noundef %A) local_unnamed_addr #0 !dbg !10 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !16, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata ptr %A, metadata !17, metadata !DIExpression()), !dbg !20
  call void @llvm.dbg.value(metadata i32 1, metadata !18, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i32 1, metadata !18, metadata !DIExpression()), !dbg !21
  %cmp.not18 = icmp slt i32 %n, 4, !dbg !22
  br i1 %cmp.not18, label %for.cond.cleanup, label %for.body.preheader, !dbg !24

for.body.preheader:                               ; preds = %entry
  %sub = add nsw i32 %n, -3
  %0 = zext i32 %sub to i64, !dbg !24
  br label %for.body, !dbg !24

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void, !dbg !25

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 1, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !18, metadata !DIExpression()), !dbg !21
  %1 = add nsw i64 %indvars.iv, -1, !dbg !26
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %1, !dbg !28
  %2 = load i32, ptr %arrayidx, align 4, !dbg !28, !tbaa !29
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv, !dbg !33
  store i32 %2, ptr %arrayidx3, align 4, !dbg !34, !tbaa !29
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 3, !dbg !35
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv.next, !dbg !36
  %3 = load i32, ptr %arrayidx5, align 4, !dbg !36, !tbaa !29
  %4 = add nuw nsw i64 %indvars.iv, 1, !dbg !37
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i64 %4, !dbg !38
  store i32 %3, ptr %arrayidx8, align 4, !dbg !39, !tbaa !29
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next, metadata !18, metadata !DIExpression()), !dbg !21
  %cmp.not = icmp ugt i64 %indvars.iv.next, %0, !dbg !22
  br i1 %cmp.not, label %for.cond.cleanup, label %for.body, !dbg !24, !llvm.loop !40
}

; // Dependence::ForwardButPreventsForwarding
; // Loop does not get vectorized despite only having a forward
; // dependency between A[i] and A[i-3].
; // This is because the store-to-load forwarding distance (here 3)
; // needs to be a multiple of vector factor otherwise the
; // store (A[5:6] in i=5) and load (A[4:5],A[6:7] in i=7,9) are unaligned.
; void test_forwardButPreventsForwarding_dep(int n, int* A, int* B) {
;   for(int i=3; i < n; ++i) {
;     A[i] = 10;
;     B[i] = A[i-3];
;   }
; }

; CHECK:remark: source.c:10:11: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop Dependence source: &A[(i + -3)] Dependence destination: &A[(i + 1)]
; CHECK-NEXT:  Backward loop carried data dependence. Memory location is the same as accessed at source.c:11:13

define dso_local void @test_forwardButPreventsForwarding_dep(i32 noundef %n, ptr nocapture noundef %A, ptr nocapture noundef writeonly %B) local_unnamed_addr #0 !dbg !43 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !47, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata ptr %A, metadata !48, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata ptr %B, metadata !49, metadata !DIExpression()), !dbg !52
  call void @llvm.dbg.value(metadata i32 3, metadata !50, metadata !DIExpression()), !dbg !53
  %cmp10 = icmp sgt i32 %n, 3, !dbg !54
  br i1 %cmp10, label %for.body.preheader, label %for.cond.cleanup, !dbg !56

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64, !dbg !54
  %0 = add nsw i64 %wide.trip.count, -3, !dbg !56
  %xtraiter = and i64 %0, 1, !dbg !56
  %1 = icmp eq i32 %n, 4, !dbg !56
  br i1 %1, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body.preheader.new, !dbg !56

for.body.preheader.new:                           ; preds = %for.body.preheader
  %unroll_iter = and i64 %0, -2, !dbg !56
  br label %for.body, !dbg !56

for.cond.cleanup.loopexit.unr-lcssa:              ; preds = %for.body, %for.body.preheader
  %indvars.iv.unr = phi i64 [ 3, %for.body.preheader ], [ %indvars.iv.next.1, %for.body ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0, !dbg !56
  br i1 %lcmp.mod.not, label %for.cond.cleanup, label %for.body.epil, !dbg !56

for.body.epil:                                    ; preds = %for.cond.cleanup.loopexit.unr-lcssa
  call void @llvm.dbg.value(metadata i64 %indvars.iv.unr, metadata !50, metadata !DIExpression()), !dbg !53
  %arrayidx.epil = getelementptr inbounds i32, ptr %A, i64 %indvars.iv.unr, !dbg !57
  store i32 10, ptr %arrayidx.epil, align 4, !dbg !59, !tbaa !29
  %2 = add nsw i64 %indvars.iv.unr, -3, !dbg !60
  %arrayidx2.epil = getelementptr inbounds i32, ptr %A, i64 %2, !dbg !61
  %3 = load i32, ptr %arrayidx2.epil, align 4, !dbg !61, !tbaa !29
  %arrayidx4.epil = getelementptr inbounds i32, ptr %B, i64 %indvars.iv.unr, !dbg !62
  store i32 %3, ptr %arrayidx4.epil, align 4, !dbg !63, !tbaa !29
  call void @llvm.dbg.value(metadata i64 %indvars.iv.unr, metadata !50, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !53
  br label %for.cond.cleanup, !dbg !64

for.cond.cleanup:                                 ; preds = %for.body.epil, %for.cond.cleanup.loopexit.unr-lcssa, %entry
  ret void, !dbg !64

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %indvars.iv = phi i64 [ 3, %for.body.preheader.new ], [ %indvars.iv.next.1, %for.body ]
  %niter = phi i64 [ 0, %for.body.preheader.new ], [ %niter.next.1, %for.body ]
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !50, metadata !DIExpression()), !dbg !53
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv, !dbg !57
  store i32 10, ptr %arrayidx, align 4, !dbg !59, !tbaa !29
  %4 = add nsw i64 %indvars.iv, -3, !dbg !60
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %4, !dbg !61
  %5 = load i32, ptr %arrayidx2, align 4, !dbg !61, !tbaa !29
  %arrayidx4 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv, !dbg !62
  store i32 %5, ptr %arrayidx4, align 4, !dbg !63, !tbaa !29
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !65
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next, metadata !50, metadata !DIExpression()), !dbg !53
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next, metadata !50, metadata !DIExpression()), !dbg !53
  %arrayidx.1 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv.next, !dbg !57
  store i32 10, ptr %arrayidx.1, align 4, !dbg !59, !tbaa !29
  %6 = add nsw i64 %indvars.iv, -2, !dbg !60
  %arrayidx2.1 = getelementptr inbounds i32, ptr %A, i64 %6, !dbg !61
  %7 = load i32, ptr %arrayidx2.1, align 4, !dbg !61, !tbaa !29
  %arrayidx4.1 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv.next, !dbg !62
  store i32 %7, ptr %arrayidx4.1, align 4, !dbg !63, !tbaa !29
  %indvars.iv.next.1 = add nuw nsw i64 %indvars.iv, 2, !dbg !65
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next.1, metadata !50, metadata !DIExpression()), !dbg !53
  %niter.next.1 = add i64 %niter, 2, !dbg !56
  %niter.ncmp.1 = icmp eq i64 %niter.next.1, %unroll_iter, !dbg !56
  br i1 %niter.ncmp.1, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body, !dbg !56, !llvm.loop !66
}

; // Dependence::BackwardVectorizableButPreventsForwarding
; // Loop does not get vectorized despite having a backward
; // but vectorizable dependency between A[i] and A[i-15].
; //
; // This is because the store-to-load forwarding distance (here 15)
; // needs to be a multiple of vector factor otherwise
; // store (A[16:17] in i=16) and load (A[15:16], A[17:18] in i=30,32) are unaligned.
; void test_backwardVectorizableButPreventsForwarding(int n, int* A) {
;   for(int i=15; i < n; ++i) {
;     A[i] = A[i-2] + A[i-15];
;   }
; }

; CHECK:remark: source.c:17:11: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop Dependence source: &A[(i + -1)] Dependence destination: &A[(i + 1)]
; CHECK: Backward loop carried data dependence. Memory location is the same as accessed at source.c:17:13

define dso_local void @test_backwardVectorizableButPreventsForwarding(i32 noundef %n, ptr nocapture noundef %A) local_unnamed_addr #0 !dbg !68 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !70, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata ptr %A, metadata !71, metadata !DIExpression()), !dbg !74
  call void @llvm.dbg.value(metadata i32 15, metadata !72, metadata !DIExpression()), !dbg !75
  %cmp12 = icmp sgt i32 %n, 15, !dbg !76
  br i1 %cmp12, label %for.body.preheader, label %for.cond.cleanup, !dbg !78

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64, !dbg !76
  %0 = add nsw i64 %wide.trip.count, -15, !dbg !78
  %xtraiter = and i64 %0, 1, !dbg !78
  %1 = icmp eq i32 %n, 16, !dbg !78
  br i1 %1, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body.preheader.new, !dbg !78

for.body.preheader.new:                           ; preds = %for.body.preheader
  %unroll_iter = and i64 %0, -2, !dbg !78
  br label %for.body, !dbg !78

for.cond.cleanup.loopexit.unr-lcssa:              ; preds = %for.body, %for.body.preheader
  %indvars.iv.unr = phi i64 [ 15, %for.body.preheader ], [ %indvars.iv.next.1, %for.body ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0, !dbg !78
  br i1 %lcmp.mod.not, label %for.cond.cleanup, label %for.body.epil, !dbg !78

for.body.epil:                                    ; preds = %for.cond.cleanup.loopexit.unr-lcssa
  call void @llvm.dbg.value(metadata i64 %indvars.iv.unr, metadata !72, metadata !DIExpression()), !dbg !75
  %2 = add nsw i64 %indvars.iv.unr, -2, !dbg !79
  %arrayidx.epil = getelementptr inbounds i32, ptr %A, i64 %2, !dbg !81
  %3 = load i32, ptr %arrayidx.epil, align 4, !dbg !81, !tbaa !29
  %4 = add nsw i64 %indvars.iv.unr, -15, !dbg !82
  %arrayidx3.epil = getelementptr inbounds i32, ptr %A, i64 %4, !dbg !83
  %5 = load i32, ptr %arrayidx3.epil, align 4, !dbg !83, !tbaa !29
  %add.epil = add nsw i32 %5, %3, !dbg !84
  %arrayidx5.epil = getelementptr inbounds i32, ptr %A, i64 %indvars.iv.unr, !dbg !85
  store i32 %add.epil, ptr %arrayidx5.epil, align 4, !dbg !86, !tbaa !29
  call void @llvm.dbg.value(metadata i64 %indvars.iv.unr, metadata !72, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !75
  br label %for.cond.cleanup, !dbg !87

for.cond.cleanup:                                 ; preds = %for.body.epil, %for.cond.cleanup.loopexit.unr-lcssa, %entry
  ret void, !dbg !87

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %indvars.iv = phi i64 [ 15, %for.body.preheader.new ], [ %indvars.iv.next.1, %for.body ]
  %niter = phi i64 [ 0, %for.body.preheader.new ], [ %niter.next.1, %for.body ]
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !72, metadata !DIExpression()), !dbg !75
  %6 = add nsw i64 %indvars.iv, -2, !dbg !79
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %6, !dbg !81
  %7 = load i32, ptr %arrayidx, align 4, !dbg !81, !tbaa !29
  %8 = add nsw i64 %indvars.iv, -15, !dbg !82
  %arrayidx3 = getelementptr inbounds i32, ptr %A, i64 %8, !dbg !83
  %9 = load i32, ptr %arrayidx3, align 4, !dbg !83, !tbaa !29
  %add = add nsw i32 %9, %7, !dbg !84
  %arrayidx5 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv, !dbg !85
  store i32 %add, ptr %arrayidx5, align 4, !dbg !86, !tbaa !29
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1, !dbg !88
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next, metadata !72, metadata !DIExpression()), !dbg !75
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next, metadata !72, metadata !DIExpression()), !dbg !75
  %10 = add nsw i64 %indvars.iv, -1, !dbg !79
  %arrayidx.1 = getelementptr inbounds i32, ptr %A, i64 %10, !dbg !81
  %11 = load i32, ptr %arrayidx.1, align 4, !dbg !81, !tbaa !29
  %12 = add nsw i64 %indvars.iv, -14, !dbg !82
  %arrayidx3.1 = getelementptr inbounds i32, ptr %A, i64 %12, !dbg !83
  %13 = load i32, ptr %arrayidx3.1, align 4, !dbg !83, !tbaa !29
  %add.1 = add nsw i32 %13, %11, !dbg !84
  %arrayidx5.1 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv.next, !dbg !85
  store i32 %add.1, ptr %arrayidx5.1, align 4, !dbg !86, !tbaa !29
  %indvars.iv.next.1 = add nuw nsw i64 %indvars.iv, 2, !dbg !88
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next.1, metadata !72, metadata !DIExpression()), !dbg !75
  %niter.next.1 = add i64 %niter, 2, !dbg !78
  %niter.ncmp.1 = icmp eq i64 %niter.next.1, %unroll_iter, !dbg !78
  br i1 %niter.ncmp.1, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body, !dbg !78, !llvm.loop !89
}

; // Dependence::Unknown
; // Different stride lengths
; void test_unknown_dep(int n, int* A) {
;   for(int i=0; i < n; ++i) {
;       A[(i+1)*4] = 10;
;       A[i] = 100;
;   }
; }

; CHECK:remark: source.c:24:13: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop Dependence source: &A[((i + 1) << 2)] Dependence destination: &A[i]
; CHECK:  Unknown data dependence. Memory location is the same as accessed at source.c:23:8

define dso_local void @test_unknown_dep(i32 noundef %n, ptr nocapture noundef writeonly %A) local_unnamed_addr #1 !dbg !91 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !93, metadata !DIExpression()), !dbg !97
  call void @llvm.dbg.value(metadata ptr %A, metadata !94, metadata !DIExpression()), !dbg !97
  call void @llvm.dbg.value(metadata i32 0, metadata !95, metadata !DIExpression()), !dbg !98
  %cmp7 = icmp sgt i32 %n, 0, !dbg !99
  br i1 %cmp7, label %for.body.preheader, label %for.cond.cleanup, !dbg !101

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64, !dbg !99
  %xtraiter = and i64 %wide.trip.count, 3, !dbg !101
  %0 = icmp ult i32 %n, 4, !dbg !101
  br i1 %0, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body.preheader.new, !dbg !101

for.body.preheader.new:                           ; preds = %for.body.preheader
  %unroll_iter = and i64 %wide.trip.count, 4294967292, !dbg !101
  br label %for.body, !dbg !101

for.cond.cleanup.loopexit.unr-lcssa:              ; preds = %for.body, %for.body.preheader
  %indvars.iv.unr = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next.3, %for.body ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0, !dbg !101
  br i1 %lcmp.mod.not, label %for.cond.cleanup, label %for.body.epil, !dbg !101

for.body.epil:                                    ; preds = %for.cond.cleanup.loopexit.unr-lcssa, %for.body.epil
  %indvars.iv.epil = phi i64 [ %indvars.iv.next.epil, %for.body.epil ], [ %indvars.iv.unr, %for.cond.cleanup.loopexit.unr-lcssa ]
  %epil.iter = phi i64 [ %epil.iter.next, %for.body.epil ], [ 0, %for.cond.cleanup.loopexit.unr-lcssa ]
  call void @llvm.dbg.value(metadata i64 %indvars.iv.epil, metadata !95, metadata !DIExpression()), !dbg !98
  %indvars.iv.next.epil = add nuw nsw i64 %indvars.iv.epil, 1, !dbg !102
  %1 = shl nsw i64 %indvars.iv.next.epil, 2, !dbg !104
  %arrayidx.epil = getelementptr inbounds i32, ptr %A, i64 %1, !dbg !105
  store i32 10, ptr %arrayidx.epil, align 4, !dbg !106, !tbaa !29
  %arrayidx2.epil = getelementptr inbounds i32, ptr %A, i64 %indvars.iv.epil, !dbg !107
  store i32 100, ptr %arrayidx2.epil, align 4, !dbg !108, !tbaa !29
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next.epil, metadata !95, metadata !DIExpression()), !dbg !98
  %epil.iter.next = add i64 %epil.iter, 1, !dbg !101
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter, !dbg !101
  br i1 %epil.iter.cmp.not, label %for.cond.cleanup, label %for.body.epil, !dbg !101, !llvm.loop !109

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit.unr-lcssa, %for.body.epil, %entry
  ret void, !dbg !111

for.body:                                         ; preds = %for.body, %for.body.preheader.new
  %indvars.iv = phi i64 [ 0, %for.body.preheader.new ], [ %indvars.iv.next.3, %for.body ]
  %niter = phi i64 [ 0, %for.body.preheader.new ], [ %niter.next.3, %for.body ]
  call void @llvm.dbg.value(metadata i64 %indvars.iv, metadata !95, metadata !DIExpression()), !dbg !98
  %indvars.iv.next = or i64 %indvars.iv, 1, !dbg !102
  %2 = shl nsw i64 %indvars.iv.next, 2, !dbg !104
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %2, !dbg !105
  store i32 10, ptr %arrayidx, align 4, !dbg !106, !tbaa !29
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv, !dbg !107
  store i32 100, ptr %arrayidx2, align 4, !dbg !108, !tbaa !29
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next, metadata !95, metadata !DIExpression()), !dbg !98
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next, metadata !95, metadata !DIExpression()), !dbg !98
  %indvars.iv.next.1 = or i64 %indvars.iv, 2, !dbg !102
  %3 = shl nsw i64 %indvars.iv.next.1, 2, !dbg !104
  %arrayidx.1 = getelementptr inbounds i32, ptr %A, i64 %3, !dbg !105
  store i32 10, ptr %arrayidx.1, align 4, !dbg !106, !tbaa !29
  %arrayidx2.1 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv.next, !dbg !107
  store i32 100, ptr %arrayidx2.1, align 4, !dbg !108, !tbaa !29
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next.1, metadata !95, metadata !DIExpression()), !dbg !98
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next.1, metadata !95, metadata !DIExpression()), !dbg !98
  %indvars.iv.next.2 = or i64 %indvars.iv, 3, !dbg !102
  %4 = shl nsw i64 %indvars.iv.next.2, 2, !dbg !104
  %arrayidx.2 = getelementptr inbounds i32, ptr %A, i64 %4, !dbg !105
  store i32 10, ptr %arrayidx.2, align 4, !dbg !106, !tbaa !29
  %arrayidx2.2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv.next.1, !dbg !107
  store i32 100, ptr %arrayidx2.2, align 4, !dbg !108, !tbaa !29
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next.2, metadata !95, metadata !DIExpression()), !dbg !98
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next.2, metadata !95, metadata !DIExpression()), !dbg !98
  %indvars.iv.next.3 = add nuw nsw i64 %indvars.iv, 4, !dbg !102
  %5 = shl nsw i64 %indvars.iv.next.3, 2, !dbg !104
  %arrayidx.3 = getelementptr inbounds i32, ptr %A, i64 %5, !dbg !105
  store i32 10, ptr %arrayidx.3, align 4, !dbg !106, !tbaa !29
  %arrayidx2.3 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv.next.2, !dbg !107
  store i32 100, ptr %arrayidx2.3, align 4, !dbg !108, !tbaa !29
  call void @llvm.dbg.value(metadata i64 %indvars.iv.next.3, metadata !95, metadata !DIExpression()), !dbg !98
  %niter.next.3 = add nuw nsw i64 %niter, 4, !dbg !101
  %niter.ncmp.3 = icmp eq i64 %niter.next.3, %unroll_iter, !dbg !101
  br i1 %niter.ncmp.3, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body, !dbg !101, !llvm.loop !112
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nofree norecurse nosync nounwind memory(argmem: write) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0 (https://github.com/phyBrackets/llvm-project-1.git 3a0a540c1307821748ab1f08e457126af0fafb6d)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "source.c", directory: "/home/shivam/llvm-project-1", checksumkind: CSK_MD5, checksum: "ce6d68d4fe0715e72ef0524124388d7f")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 17.0.0 (https://github.com/phyBrackets/llvm-project-1.git 3a0a540c1307821748ab1f08e457126af0fafb6d)"}
!10 = distinct !DISubprogram(name: "test_backward_dep", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !15)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !13, !14}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!15 = !{!16, !17, !18}
!16 = !DILocalVariable(name: "n", arg: 1, scope: !10, file: !1, line: 1, type: !13)
!17 = !DILocalVariable(name: "A", arg: 2, scope: !10, file: !1, line: 1, type: !14)
!18 = !DILocalVariable(name: "i", scope: !19, file: !1, line: 2, type: !13)
!19 = distinct !DILexicalBlock(scope: !10, file: !1, line: 2, column: 4)
!20 = !DILocation(line: 0, scope: !10)
!21 = !DILocation(line: 0, scope: !19)
!22 = !DILocation(line: 2, column: 22, scope: !23)
!23 = distinct !DILexicalBlock(scope: !19, file: !1, line: 2, column: 4)
!24 = !DILocation(line: 2, column: 4, scope: !19)
!25 = !DILocation(line: 6, column: 2, scope: !10)
!26 = !DILocation(line: 3, column: 16, scope: !27)
!27 = distinct !DILexicalBlock(scope: !23, file: !1, line: 2, column: 40)
!28 = !DILocation(line: 3, column: 13, scope: !27)
!29 = !{!30, !30, i64 0}
!30 = !{!"int", !31, i64 0}
!31 = !{!"omnipotent char", !32, i64 0}
!32 = !{!"Simple C/C++ TBAA"}
!33 = !DILocation(line: 3, column: 6, scope: !27)
!34 = !DILocation(line: 3, column: 11, scope: !27)
!35 = !DILocation(line: 4, column: 17, scope: !27)
!36 = !DILocation(line: 4, column: 14, scope: !27)
!37 = !DILocation(line: 4, column: 8, scope: !27)
!38 = !DILocation(line: 4, column: 5, scope: !27)
!39 = !DILocation(line: 4, column: 12, scope: !27)
!40 = distinct !{!40, !24, !41, !42}
!41 = !DILocation(line: 5, column: 4, scope: !19)
!42 = !{!"llvm.loop.mustprogress"}
!43 = distinct !DISubprogram(name: "test_forwardButPreventsForwarding_dep", scope: !1, file: !1, line: 8, type: !44, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !46)
!44 = !DISubroutineType(types: !45)
!45 = !{null, !13, !14, !14}
!46 = !{!47, !48, !49, !50}
!47 = !DILocalVariable(name: "n", arg: 1, scope: !43, file: !1, line: 8, type: !13)
!48 = !DILocalVariable(name: "A", arg: 2, scope: !43, file: !1, line: 8, type: !14)
!49 = !DILocalVariable(name: "B", arg: 3, scope: !43, file: !1, line: 8, type: !14)
!50 = !DILocalVariable(name: "i", scope: !51, file: !1, line: 9, type: !13)
!51 = distinct !DILexicalBlock(scope: !43, file: !1, line: 9, column: 4)
!52 = !DILocation(line: 0, scope: !43)
!53 = !DILocation(line: 0, scope: !51)
!54 = !DILocation(line: 9, column: 19, scope: !55)
!55 = distinct !DILexicalBlock(scope: !51, file: !1, line: 9, column: 4)
!56 = !DILocation(line: 9, column: 4, scope: !51)
!57 = !DILocation(line: 10, column: 6, scope: !58)
!58 = distinct !DILexicalBlock(scope: !55, file: !1, line: 9, column: 29)
!59 = !DILocation(line: 10, column: 11, scope: !58)
!60 = !DILocation(line: 11, column: 16, scope: !58)
!61 = !DILocation(line: 11, column: 13, scope: !58)
!62 = !DILocation(line: 11, column: 6, scope: !58)
!63 = !DILocation(line: 11, column: 11, scope: !58)
!64 = !DILocation(line: 13, column: 2, scope: !43)
!65 = !DILocation(line: 9, column: 24, scope: !55)
!66 = distinct !{!66, !56, !67, !42}
!67 = !DILocation(line: 12, column: 4, scope: !51)
!68 = distinct !DISubprogram(name: "test_backwardVectorizableButPreventsForwarding", scope: !1, file: !1, line: 15, type: !11, scopeLine: 15, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !69)
!69 = !{!70, !71, !72}
!70 = !DILocalVariable(name: "n", arg: 1, scope: !68, file: !1, line: 15, type: !13)
!71 = !DILocalVariable(name: "A", arg: 2, scope: !68, file: !1, line: 15, type: !14)
!72 = !DILocalVariable(name: "i", scope: !73, file: !1, line: 16, type: !13)
!73 = distinct !DILexicalBlock(scope: !68, file: !1, line: 16, column: 4)
!74 = !DILocation(line: 0, scope: !68)
!75 = !DILocation(line: 0, scope: !73)
!76 = !DILocation(line: 16, column: 20, scope: !77)
!77 = distinct !DILexicalBlock(scope: !73, file: !1, line: 16, column: 4)
!78 = !DILocation(line: 16, column: 4, scope: !73)
!79 = !DILocation(line: 17, column: 16, scope: !80)
!80 = distinct !DILexicalBlock(scope: !77, file: !1, line: 16, column: 30)
!81 = !DILocation(line: 17, column: 13, scope: !80)
!82 = !DILocation(line: 17, column: 25, scope: !80)
!83 = !DILocation(line: 17, column: 22, scope: !80)
!84 = !DILocation(line: 17, column: 20, scope: !80)
!85 = !DILocation(line: 17, column: 6, scope: !80)
!86 = !DILocation(line: 17, column: 11, scope: !80)
!87 = !DILocation(line: 19, column: 2, scope: !68)
!88 = !DILocation(line: 16, column: 25, scope: !77)
!89 = distinct !{!89, !78, !90, !42}
!90 = !DILocation(line: 18, column: 4, scope: !73)
!91 = distinct !DISubprogram(name: "test_unknown_dep", scope: !1, file: !1, line: 21, type: !11, scopeLine: 21, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !92)
!92 = !{!93, !94, !95}
!93 = !DILocalVariable(name: "n", arg: 1, scope: !91, file: !1, line: 21, type: !13)
!94 = !DILocalVariable(name: "A", arg: 2, scope: !91, file: !1, line: 21, type: !14)
!95 = !DILocalVariable(name: "i", scope: !96, file: !1, line: 22, type: !13)
!96 = distinct !DILexicalBlock(scope: !91, file: !1, line: 22, column: 4)
!97 = !DILocation(line: 0, scope: !91)
!98 = !DILocation(line: 0, scope: !96)
!99 = !DILocation(line: 22, column: 19, scope: !100)
!100 = distinct !DILexicalBlock(scope: !96, file: !1, line: 22, column: 4)
!101 = !DILocation(line: 22, column: 4, scope: !96)
!102 = !DILocation(line: 23, column: 12, scope: !103)
!103 = distinct !DILexicalBlock(scope: !100, file: !1, line: 22, column: 29)
!104 = !DILocation(line: 23, column: 15, scope: !103)
!105 = !DILocation(line: 23, column: 8, scope: !103)
!106 = !DILocation(line: 23, column: 19, scope: !103)
!107 = !DILocation(line: 24, column: 8, scope: !103)
!108 = !DILocation(line: 24, column: 13, scope: !103)
!109 = distinct !{!109, !110}
!110 = !{!"llvm.loop.unroll.disable"}
!111 = !DILocation(line: 26, column: 2, scope: !91)
!112 = distinct !{!112, !101, !113, !42}
!113 = !DILocation(line: 25, column: 4, scope: !96)
