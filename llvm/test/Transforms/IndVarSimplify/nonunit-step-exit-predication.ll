; RUN: opt -passes=indvars -S < %s | FileCheck %s

;; These non-unit-step extension to ScalarEvolution,
;;
;; Tests 1, 3, 4: symbolic bounds with nuw/nsw flags.
;; Tests 2, 5: negative tests — no nuw/nsw flags, check stays in loop.
;; Test 6: constant bound.

;; ============================================================
;; Test 1: step +8, nuw, unsigned < — bounds check hoisted to preheader
;; ============================================================
define i32 @step8_nuw_ult(i32 %start, i32 %bound, i32 %n) {
; CHECK-LABEL: @step8_nuw_ult(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i32 @llvm.umax.i32(i32 %bound, i32 %start)
; CHECK-NEXT:    [[TMP0:%.*]] = add i32 [[UMAX]], 7
; CHECK-NEXT:    [[TMP1:%.*]] = sub i32 [[TMP0]], %start
; CHECK-NEXT:    [[TMP2:%.*]] = lshr i32 [[TMP1]], 3
; CHECK-NEXT:    [[UMAX2:%.*]] = call i32 @llvm.umax.i32(i32 %n, i32 1)
; CHECK-NEXT:    [[TMP3:%.*]] = add i32 [[UMAX2]], -1
; CHECK-NEXT:    [[TMP4:%.*]] = freeze i32 [[TMP3]]
; CHECK-NEXT:    [[UMIN:%.*]] = call i32 @llvm.umin.i32(i32 [[TMP4]], i32 [[TMP2]])
; CHECK-NEXT:    [[TMP5:%.*]] = icmp ne i32 [[TMP2]], [[UMIN]]
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ %start, %entry ], [ [[IV_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ 0, %entry ], [ [[I_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    br i1 [[TMP5]], label %backedge, label %failed
; CHECK:       backedge:
; CHECK-NEXT:    [[IV_NEXT]] = add nuw i32 [[IV]], 8
; CHECK-NEXT:    [[I_NEXT]] = add nuw nsw i32 [[I]], 1
; CHECK-NEXT:    [[LOOP_COND:%.*]] = icmp ult i32 [[I_NEXT]], %n
; CHECK-NEXT:    br i1 [[LOOP_COND]], label %loop, label %done
; CHECK:       done:
; CHECK-NEXT:    [[IV_LCSSA1:%.*]] = phi i32 [ [[IV]], %backedge ]
; CHECK-NEXT:    ret i32 [[IV_LCSSA1]]
; CHECK:       failed:
; CHECK-NEXT:    ret i32 -1
;
entry:
  br label %loop
loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %backedge]
  %i = phi i32 [0, %entry], [%i.next, %backedge]
  %check = icmp ult i32 %iv, %bound
  br i1 %check, label %backedge, label %failed
backedge:
  %iv.next = add nuw i32 %iv, 8
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %done
done:
  ret i32 %iv
failed:
  ret i32 -1
}

;; ============================================================
;; Test 2: step +8, NO nuw, unsigned
;; ============================================================
define i32 @step8_no_nuw_ult(i32 %start, i32 %bound, i32 %n) {
; CHECK-LABEL: @step8_no_nuw_ult(
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ %start, %entry ], [ [[IV_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ 0, %entry ], [ [[I_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    [[CHECK:%.*]] = icmp ult i32 [[IV]], %bound
; CHECK-NEXT:    br i1 [[CHECK]], label %backedge, label %failed
;
entry:
  br label %loop
loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %backedge]
  %i = phi i32 [0, %entry], [%i.next, %backedge]
  %check = icmp ult i32 %iv, %bound
  br i1 %check, label %backedge, label %failed
backedge:
  %iv.next = add i32 %iv, 8
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %done
done:
  ret i32 %iv
failed:
  ret i32 -1
}

;; ============================================================
;; Test 3: step +8, nuw+nsw, signed
;; ============================================================
define i32 @step8_nuw_nsw_slt(i32 %start, i32 %bound, i32 %n) {
; CHECK-LABEL: @step8_nuw_nsw_slt(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[SMAX:%.*]] = call i32 @llvm.smax.i32(i32 %bound, i32 %start)
; CHECK-NEXT:    [[TMP0:%.*]] = add i32 [[SMAX]], 7
; CHECK-NEXT:    [[TMP1:%.*]] = sub i32 [[TMP0]], %start
; CHECK-NEXT:    [[TMP2:%.*]] = lshr i32 [[TMP1]], 3
; CHECK-NEXT:    [[UMAX:%.*]] = call i32 @llvm.umax.i32(i32 %n, i32 1)
; CHECK-NEXT:    [[TMP3:%.*]] = add i32 [[UMAX]], -1
; CHECK-NEXT:    [[TMP4:%.*]] = freeze i32 [[TMP3]]
; CHECK-NEXT:    [[UMIN:%.*]] = call i32 @llvm.umin.i32(i32 [[TMP4]], i32 [[TMP2]])
; CHECK-NEXT:    [[TMP5:%.*]] = icmp ne i32 [[TMP2]], [[UMIN]]
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ %start, %entry ], [ [[IV_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ 0, %entry ], [ [[I_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    br i1 [[TMP5]], label %backedge, label %failed
;
entry:
  br label %loop
loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %backedge]
  %i = phi i32 [0, %entry], [%i.next, %backedge]
  %check = icmp slt i32 %iv, %bound
  br i1 %check, label %backedge, label %failed
backedge:
  %iv.next = add nuw nsw i32 %iv, 8
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %done
done:
  ret i32 %iv
failed:
  ret i32 -1
}

;; ============================================================
;; Test 4: step -4, nsw, signed > — bounds check hoisted to preheader
;; ============================================================
define i32 @neg_step4_nsw_sgt(i32 %start, i32 %bound, i32 %n) {
; CHECK-LABEL: @neg_step4_nsw_sgt(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = add i32 %start, 3
; CHECK-NEXT:    [[SMIN:%.*]] = call i32 @llvm.smin.i32(i32 %bound, i32 %start)
; CHECK-NEXT:    [[TMP1:%.*]] = sub i32 [[TMP0]], [[SMIN]]
; CHECK-NEXT:    [[TMP2:%.*]] = lshr i32 [[TMP1]], 2
; CHECK-NEXT:    [[UMAX:%.*]] = call i32 @llvm.umax.i32(i32 %n, i32 1)
; CHECK-NEXT:    [[TMP3:%.*]] = add i32 [[UMAX]], -1
; CHECK-NEXT:    [[TMP4:%.*]] = freeze i32 [[TMP3]]
; CHECK-NEXT:    [[UMIN:%.*]] = call i32 @llvm.umin.i32(i32 [[TMP4]], i32 [[TMP2]])
; CHECK-NEXT:    [[TMP5:%.*]] = icmp ne i32 [[TMP2]], [[UMIN]]
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ %start, %entry ], [ [[IV_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ 0, %entry ], [ [[I_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    br i1 [[TMP5]], label %backedge, label %failed
;
entry:
  br label %loop
loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %backedge]
  %i = phi i32 [0, %entry], [%i.next, %backedge]
  %check = icmp sgt i32 %iv, %bound
  br i1 %check, label %backedge, label %failed
backedge:
  %iv.next = add nsw i32 %iv, -4
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %done
done:
  ret i32 %iv
failed:
  ret i32 -1
}

;; ============================================================
;; Test 5: step -4, NO nsw, signed > — check stays in loop
;; ============================================================
define i32 @neg_step4_no_nsw_sgt(i32 %start, i32 %bound, i32 %n) {
; CHECK-LABEL: @neg_step4_no_nsw_sgt(
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ %start, %entry ], [ [[IV_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ 0, %entry ], [ [[I_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    [[CHECK:%.*]] = icmp sgt i32 [[IV]], %bound
; CHECK-NEXT:    br i1 [[CHECK]], label %backedge, label %failed
;
entry:
  br label %loop
loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %backedge]
  %i = phi i32 [0, %entry], [%i.next, %backedge]
  %check = icmp sgt i32 %iv, %bound
  br i1 %check, label %backedge, label %failed
backedge:
  %iv.next = add i32 %iv, -4
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %done
done:
  ret i32 %iv
failed:
  ret i32 -1
}

;; ============================================================
;; Test 6: step +8, nuw, constant bound — DOES hoist via
;; computable exit count (canIVOverflowOnLT returns false for
;; constant bound <= UINT_MAX - stride + 1)
;; ============================================================
define i32 @step8_nuw_const_bound(i32 %start, i32 %n) {
; CHECK-LABEL: @step8_nuw_const_bound(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i32 @llvm.umax.i32(i32 %start, i32 1000)
; CHECK-NEXT:    [[TMP0:%.*]] = add i32 [[UMAX]], 7
; CHECK-NEXT:    [[TMP1:%.*]] = sub i32 [[TMP0]], %start
; CHECK-NEXT:    [[TMP2:%.*]] = lshr i32 [[TMP1]], 3
; CHECK-NEXT:    [[UMAX2:%.*]] = call i32 @llvm.umax.i32(i32 %n, i32 1)
; CHECK-NEXT:    [[TMP3:%.*]] = add i32 [[UMAX2]], -1
; CHECK-NEXT:    [[TMP4:%.*]] = freeze i32 [[TMP3]]
; CHECK-NEXT:    [[UMIN:%.*]] = call i32 @llvm.umin.i32(i32 [[TMP4]], i32 [[TMP2]])
; CHECK-NEXT:    [[TMP5:%.*]] = icmp ne i32 [[TMP2]], [[UMIN]]
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ %start, %entry ], [ [[IV_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ 0, %entry ], [ [[I_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    br i1 [[TMP5]], label %backedge, label %failed
; CHECK:       backedge:
; CHECK-NEXT:    [[IV_NEXT]] = add nuw nsw i32 [[IV]], 8
; CHECK-NEXT:    [[I_NEXT]] = add nuw nsw i32 [[I]], 1
; CHECK-NEXT:    [[LOOP_COND:%.*]] = icmp ult i32 [[I_NEXT]], %n
; CHECK-NEXT:    br i1 [[LOOP_COND]], label %loop, label %done
; CHECK:       done:
; CHECK-NEXT:    [[IV_LCSSA2:%.*]] = phi i32 [ [[IV]], %backedge ]
; CHECK-NEXT:    ret i32 [[IV_LCSSA2]]
; CHECK:       failed:
; CHECK-NEXT:    ret i32 -1
;
entry:
  br label %loop
loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %backedge]
  %i = phi i32 [0, %entry], [%i.next, %backedge]
  %check = icmp ult i32 %iv, 1000
  br i1 %check, label %backedge, label %failed
backedge:
  %iv.next = add nuw i32 %iv, 8
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %done
done:
  ret i32 %iv
failed:
  ret i32 -1
}

;; ============================================================
;; Boundary condition tests
;; ============================================================

;; ============================================================
;; Test 7: step 1, nuw, constant bound — regression test
;; Step=1 should still hoist via computable exit count path
;; ============================================================
define i32 @step1_nuw_ult_const(i32 %start, i32 %n) {
; CHECK-LABEL: @step1_nuw_ult_const(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i32 @llvm.umax.i32(i32 %start, i32 1000)
; CHECK-NEXT:    [[TMP0:%.*]] = sub i32 [[UMAX]], %start
; CHECK-NEXT:    [[UMAX2:%.*]] = call i32 @llvm.umax.i32(i32 %n, i32 1)
; CHECK-NEXT:    [[TMP1:%.*]] = add i32 [[UMAX2]], -1
; CHECK-NEXT:    [[TMP2:%.*]] = freeze i32 [[TMP1]]
; CHECK-NEXT:    [[UMIN:%.*]] = call i32 @llvm.umin.i32(i32 [[TMP2]], i32 [[TMP0]])
; CHECK-NEXT:    [[TMP3:%.*]] = icmp ne i32 [[TMP0]], [[UMIN]]
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ %start, %entry ], [ [[IV_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ 0, %entry ], [ [[I_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    br i1 [[TMP3]], label %backedge, label %failed
;
entry:
  br label %loop
loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %backedge]
  %i = phi i32 [0, %entry], [%i.next, %backedge]
  %check = icmp ult i32 %iv, 1000
  br i1 %check, label %backedge, label %failed
backedge:
  %iv.next = add nuw i32 %iv, 1
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %done
done:
  ret i32 %iv
failed:
  ret i32 -1
}

;; ============================================================
;; Test 8: step 2 (smallest non-unit), nuw, constant bound — should hoist
;; ============================================================
define i32 @step2_nuw_ult_const(i32 %start, i32 %n) {
; CHECK-LABEL: @step2_nuw_ult_const(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i32 @llvm.umax.i32(i32 %start, i32 500)
; CHECK-NEXT:    [[TMP0:%.*]] = add i32 [[UMAX]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = sub i32 [[TMP0]], %start
; CHECK-NEXT:    [[TMP2:%.*]] = lshr i32 [[TMP1]], 1
; CHECK-NEXT:    [[UMAX2:%.*]] = call i32 @llvm.umax.i32(i32 %n, i32 1)
; CHECK-NEXT:    [[TMP3:%.*]] = add i32 [[UMAX2]], -1
; CHECK-NEXT:    [[TMP4:%.*]] = freeze i32 [[TMP3]]
; CHECK-NEXT:    [[UMIN:%.*]] = call i32 @llvm.umin.i32(i32 [[TMP4]], i32 [[TMP2]])
; CHECK-NEXT:    [[TMP5:%.*]] = icmp ne i32 [[TMP2]], [[UMIN]]
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ %start, %entry ], [ [[IV_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ 0, %entry ], [ [[I_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    br i1 [[TMP5]], label %backedge, label %failed
;
entry:
  br label %loop
loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %backedge]
  %i = phi i32 [0, %entry], [%i.next, %backedge]
  %check = icmp ult i32 %iv, 500
  br i1 %check, label %backedge, label %failed
backedge:
  %iv.next = add nuw i32 %iv, 2
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %done
done:
  ret i32 %iv
failed:
  ret i32 -1
}

;; ============================================================
;; Test 9: step +8, nuw, bound = UINT_MAX-6 (4294967289)
;; canIVOverflowOnLT returns true for this bound, but nuw on the
;; AddRec makes the exit count computable — should hoist
;; ============================================================
define i32 @step8_nuw_ult_near_max_bound(i32 %start, i32 %n) {
; CHECK-LABEL: @step8_nuw_ult_near_max_bound(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i32 @llvm.umax.i32(i32 %start, i32 -7)
; CHECK-NEXT:    [[TMP0:%.*]] = add i32 [[UMAX]], 7
; CHECK-NEXT:    [[TMP1:%.*]] = sub i32 [[TMP0]], %start
; CHECK-NEXT:    [[TMP2:%.*]] = lshr i32 [[TMP1]], 3
; CHECK-NEXT:    [[UMAX2:%.*]] = call i32 @llvm.umax.i32(i32 %n, i32 1)
; CHECK-NEXT:    [[TMP3:%.*]] = add i32 [[UMAX2]], -1
; CHECK-NEXT:    [[TMP4:%.*]] = freeze i32 [[TMP3]]
; CHECK-NEXT:    [[UMIN:%.*]] = call i32 @llvm.umin.i32(i32 [[TMP4]], i32 [[TMP2]])
; CHECK-NEXT:    [[TMP5:%.*]] = icmp ne i32 [[TMP2]], [[UMIN]]
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ %start, %entry ], [ [[IV_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ 0, %entry ], [ [[I_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    br i1 [[TMP5]], label %backedge, label %failed
;
entry:
  br label %loop
loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %backedge]
  %i = phi i32 [0, %entry], [%i.next, %backedge]
  %check = icmp ult i32 %iv, 4294967289
  br i1 %check, label %backedge, label %failed
backedge:
  %iv.next = add nuw i32 %iv, 8
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %done
done:
  ret i32 %iv
failed:
  ret i32 -1
}

;; ============================================================
;; Test 10: step +8, nuw, bound = 0 — icmp ult %iv, 0 is always false
;; LLVM folds to br i1 false
;; ============================================================
define i32 @step8_nuw_ult_zero_bound(i32 %start, i32 %n) {
; CHECK-LABEL: @step8_nuw_ult_zero_bound(
; CHECK:       loop:
; CHECK-NEXT:    br i1 false, label %backedge, label %failed
;
entry:
  br label %loop
loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %backedge]
  %i = phi i32 [0, %entry], [%i.next, %backedge]
  %check = icmp ult i32 %iv, 0
  br i1 %check, label %backedge, label %failed
backedge:
  %iv.next = add nuw i32 %iv, 8
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %done
done:
  ret i32 %iv
failed:
  ret i32 -1
}

;; ============================================================
;; Test 11: step +8, nuw, start=0, constant bound — should hoist
;; Known start simplifies exit count computation
;; ============================================================
define i32 @step8_nuw_ult_zero_start(i32 %n) {
; CHECK-LABEL: @step8_nuw_ult_zero_start(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i32 @llvm.umax.i32(i32 %n, i32 1)
; CHECK-NEXT:    [[TMP0:%.*]] = add i32 [[UMAX]], -1
; CHECK-NEXT:    [[UMIN:%.*]] = call i32 @llvm.umin.i32(i32 [[TMP0]], i32 125)
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ne i32 125, [[UMIN]]
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ 0, %entry ], [ [[IV_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ 0, %entry ], [ [[I_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    br i1 [[TMP1]], label %backedge, label %failed
;
entry:
  br label %loop
loop:
  %iv = phi i32 [0, %entry], [%iv.next, %backedge]
  %i = phi i32 [0, %entry], [%i.next, %backedge]
  %check = icmp ult i32 %iv, 1000
  br i1 %check, label %backedge, label %failed
backedge:
  %iv.next = add nuw i32 %iv, 8
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %done
done:
  ret i32 %iv
failed:
  ret i32 -1
}

;; ============================================================
;; Test 12: step +8, nuw, n=1 (single iteration), constant bound
;; Should hoist; backedge always exits after 1 iteration
;; ============================================================
define i32 @step8_nuw_ult_single_iter(i32 %start) {
; CHECK-LABEL: @step8_nuw_ult_single_iter(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i32 @llvm.umax.i32(i32 %start, i32 1000)
; CHECK-NEXT:    [[TMP0:%.*]] = add i32 [[UMAX]], 7
; CHECK-NEXT:    [[TMP1:%.*]] = sub i32 [[TMP0]], %start
; CHECK-NEXT:    [[TMP2:%.*]] = lshr i32 [[TMP1]], 3
; CHECK-NEXT:    [[TMP3:%.*]] = icmp ne i32 [[TMP2]], 0
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    br i1 [[TMP3]], label %backedge, label %failed
; CHECK:       backedge:
; CHECK-NEXT:    br i1 false, label %loop, label %done
; CHECK:       done:
; CHECK-NEXT:    ret i32 %start
;
entry:
  br label %loop
loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %backedge]
  %i = phi i32 [0, %entry], [%i.next, %backedge]
  %check = icmp ult i32 %iv, 1000
  br i1 %check, label %backedge, label %failed
backedge:
  %iv.next = add nuw i32 %iv, 8
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, 1
  br i1 %loop_cond, label %loop, label %done
done:
  ret i32 %iv
failed:
  ret i32 -1
}

;; ============================================================
;; Test 13: step +8, nuw, bound = UINT_MAX-7 (4294967288)
;; Exact overflow boundary: canIVOverflowOnLT(4294967288, 8, false)
;; checks (UINT_MAX-7).ult(4294967288) = 4294967288.ult(4294967288) = false
;; So exit count IS computable — should hoist
;; ============================================================
define i32 @step8_nuw_ult_exact_overflow_boundary(i32 %start, i32 %n) {
; CHECK-LABEL: @step8_nuw_ult_exact_overflow_boundary(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[UMAX:%.*]] = call i32 @llvm.umax.i32(i32 %start, i32 -8)
; CHECK-NEXT:    [[TMP0:%.*]] = add i32 [[UMAX]], 7
; CHECK-NEXT:    [[TMP1:%.*]] = sub i32 [[TMP0]], %start
; CHECK-NEXT:    [[TMP2:%.*]] = lshr i32 [[TMP1]], 3
; CHECK-NEXT:    [[UMAX2:%.*]] = call i32 @llvm.umax.i32(i32 %n, i32 1)
; CHECK-NEXT:    [[TMP3:%.*]] = add i32 [[UMAX2]], -1
; CHECK-NEXT:    [[TMP4:%.*]] = freeze i32 [[TMP3]]
; CHECK-NEXT:    [[UMIN:%.*]] = call i32 @llvm.umin.i32(i32 [[TMP4]], i32 [[TMP2]])
; CHECK-NEXT:    [[TMP5:%.*]] = icmp ne i32 [[TMP2]], [[UMIN]]
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    [[IV:%.*]] = phi i32 [ %start, %entry ], [ [[IV_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ 0, %entry ], [ [[I_NEXT:%.*]], %backedge ]
; CHECK-NEXT:    br i1 [[TMP5]], label %backedge, label %failed
;
entry:
  br label %loop
loop:
  %iv = phi i32 [%start, %entry], [%iv.next, %backedge]
  %i = phi i32 [0, %entry], [%i.next, %backedge]
  %check = icmp ult i32 %iv, 4294967288
  br i1 %check, label %backedge, label %failed
backedge:
  %iv.next = add nuw i32 %iv, 8
  %i.next = add nuw i32 %i, 1
  %loop_cond = icmp ult i32 %i.next, %n
  br i1 %loop_cond, label %loop, label %done
done:
  ret i32 %iv
failed:
  ret i32 -1
}
