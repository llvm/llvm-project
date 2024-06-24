; RUN: opt -polly-use-llvm-names -polly-region-expansion-profitability-check=1 \
; RUN: "-passes=scop(polly-opt-isl,print<polly-ast>)" -disable-output < %s | \
; RUN: FileCheck %s --check-prefix=HEURISTIC
; RUN: opt -polly-region-expansion-profitability-check=0 -passes=polly-codegen \
; RUN: -pass-remarks-analysis="polly-scops" -disable-output < %s 2>&1 | \
; RUN: FileCheck %s --check-prefix=NO_HEURISTIC

; void test(int **restrict a, int *restrict b, int *restrict c,
;           int *restrict d, int *restrict e, int *restrict f,
;	    int L, int M, int Val0) {
;   for (int i = 1; i <= L; i++) { // L1
;     for (int k = 1; k <= M; k++) { // L2: vectorizable loop.
;       b[k] += e[k-1];
;       if (k < M)
;         c[k] += d[k];
;     }
;     // Memory accesses to a[i-1],a[i] are not used in L2.
;     if ((Val0 = a[i-1][4]) > -987654321)
;      a[i][4] = Val0;
;   }
; }


target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"

define dso_local void @test(ptr noalias nocapture noundef readonly %a, ptr noalias nocapture noundef %b, ptr noalias nocapture noundef %c, ptr noalias nocapture noundef readonly %d, ptr noalias nocapture noundef readonly %e, ptr noalias nocapture noundef readnone %f, i32 noundef %L, i32 noundef %M, i32 noundef %Val0) {
entry:
  %cmp.not39 = icmp slt i32 %L, 1
  br i1 %cmp.not39, label %for.cond.cleanup, label %for.cond1.preheader.lr.ph

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %invariant.gep = getelementptr i8, ptr %e, i64 -4
  %cmp2.not37 = icmp slt i32 %M, 1
  br i1 %cmp2.not37, label %for.cond1.preheader.us.preheader, label %for.cond1.preheader.preheader

for.cond1.preheader.preheader:                    ; preds = %for.cond1.preheader.lr.ph
  %0 = zext nneg i32 %M to i64
  %1 = add nuw i32 %M, 1
  %2 = add nuw i32 %L, 1
  %wide.trip.count46 = zext i32 %2 to i64
  %wide.trip.count = zext i32 %1 to i64
  br label %for.cond1.preheader

for.cond1.preheader.us.preheader:                 ; preds = %for.cond1.preheader.lr.ph
  %3 = add nuw i32 %L, 1
  %wide.trip.count51 = zext i32 %3 to i64
  br label %for.cond1.preheader.us

for.cond1.preheader.us:                           ; preds = %for.cond1.preheader.us.preheader, %for.inc23.us
  %indvars.iv48 = phi i64 [ 1, %for.cond1.preheader.us.preheader ], [ %indvars.iv.next49, %for.inc23.us ]
  %4 = getelementptr ptr, ptr %a, i64 %indvars.iv48
  %arrayidx15.us = getelementptr i8, ptr %4, i64 -8
  %5 = load ptr, ptr %arrayidx15.us, align 8
  %arrayidx16.us = getelementptr inbounds i8, ptr %5, i64 16
  %6 = load i32, ptr %arrayidx16.us, align 4
  %cmp17.us = icmp sgt i32 %6, -987654321
  br i1 %cmp17.us, label %if.then18.us, label %for.inc23.us

if.then18.us:                                     ; preds = %for.cond1.preheader.us
  %7 = load ptr, ptr %4, align 8
  %arrayidx21.us = getelementptr inbounds i8, ptr %7, i64 16
  store i32 %6, ptr %arrayidx21.us, align 4
  br label %for.inc23.us

for.inc23.us:                                     ; preds = %if.then18.us, %for.cond1.preheader.us
  %indvars.iv.next49 = add nuw nsw i64 %indvars.iv48, 1
  %exitcond52.not = icmp eq i64 %indvars.iv.next49, %wide.trip.count51
  br i1 %exitcond52.not, label %for.cond.cleanup, label %for.cond1.preheader.us

for.cond1.preheader:                              ; preds = %for.cond1.preheader.preheader, %for.inc23
  %indvars.iv43 = phi i64 [ 1, %for.cond1.preheader.preheader ], [ %indvars.iv.next44, %for.inc23 ]
  br label %for.body4

for.cond.cleanup:                                 ; preds = %for.inc23, %for.inc23.us, %entry
  ret void

for.cond1.for.cond.cleanup3_crit_edge:            ; preds = %for.inc
  %8 = getelementptr ptr, ptr %a, i64 %indvars.iv43
  %arrayidx15 = getelementptr i8, ptr %8, i64 -8
  %9 = load ptr, ptr %arrayidx15, align 8
  %arrayidx16 = getelementptr inbounds i8, ptr %9, i64 16
  %10 = load i32, ptr %arrayidx16, align 4
  %cmp17 = icmp sgt i32 %10, -987654321
  br i1 %cmp17, label %if.then18, label %for.inc23

for.body4:                                        ; preds = %for.cond1.preheader, %for.inc
  %indvars.iv = phi i64 [ 1, %for.cond1.preheader ], [ %indvars.iv.next, %for.inc ]
  %gep = getelementptr i32, ptr %invariant.gep, i64 %indvars.iv
  %11 = load i32, ptr %gep, align 4
  %arrayidx6 = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %12 = load i32, ptr %arrayidx6, align 4
  %add = add nsw i32 %12, %11
  store i32 %add, ptr %arrayidx6, align 4
  %cmp7 = icmp ult i64 %indvars.iv, %0
  br i1 %cmp7, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body4
  %arrayidx9 = getelementptr inbounds i32, ptr %d, i64 %indvars.iv
  %13 = load i32, ptr %arrayidx9, align 4
  %arrayidx11 = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  %14 = load i32, ptr %arrayidx11, align 4
  %add12 = add nsw i32 %14, %13
  store i32 %add12, ptr %arrayidx11, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body4, %if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond1.for.cond.cleanup3_crit_edge, label %for.body4

if.then18:                                        ; preds = %for.cond1.for.cond.cleanup3_crit_edge
  %15 = load ptr, ptr %8, align 8
  %arrayidx21 = getelementptr inbounds i8, ptr %15, i64 16
  store i32 %10, ptr %arrayidx21, align 4
  br label %for.inc23

for.inc23:                                        ; preds = %for.cond1.for.cond.cleanup3_crit_edge, %if.then18
  %indvars.iv.next44 = add nuw nsw i64 %indvars.iv43, 1
  %exitcond47.not = icmp eq i64 %indvars.iv.next44, %wide.trip.count46
  br i1 %exitcond47.not, label %for.cond.cleanup, label %for.cond1.preheader
}

; Check that region %for.body4---%for.cond1.for.cond.cleanup3_crit_edge is
; detected with region expansion profitability check.
; HEURISTIC-LABLE: isl ast :: test :: %for.body4---%for.cond1.for.cond.cleanup3_crit_edge
; HEURISTIC:      for (int c0 = 0; c0 < M; c0 += 1)
; HEURISTIC-NEXT:	Stmt_for_body4(c0);
; HEURISTIC-NEXT: for (int c0 = 0; c0 < M - 1; c0 += 1)
; HEURISTIC-NEXT:	Stmt_if_then(c0);

; Check that without region expansion heuristic, vectorization is missed.
; NO_HEURISTIC: SCoP begins here.
; NO_HEURISTIC-NEXT: No-overflows restriction:        [p_0, p_1] -> {  : p_0 = 2147483647 }
; NO_HEURISTIC-NEXT: No-overflows restriction:        [p_0, p_1] -> {  : p_0 = 2147483647 }
; NO_HEURISTIC-NEXT: Possibly aliasing pointer, use restrict keyword.
; NO_HEURISTIC-NEXT: Possibly aliasing pointer, use restrict keyword.
; NO_HEURISTIC-NEXT: Invariant load assumption:  [p_0, p_1] -> {  : false }
; NO_HEURISTIC-NEXT : SCoP ends here but was dismissed.
