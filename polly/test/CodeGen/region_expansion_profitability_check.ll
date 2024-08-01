; RUN: opt %loadNPMPolly -polly-region-expansion-profitability-check=1 \
; RUN: "-passes=scop(polly-opt-isl,print<polly-ast>)" -disable-output < %s | \
; RUN: FileCheck %s --check-prefix=HEURISTIC
; RUN: opt %loadNPMPolly -polly-region-expansion-profitability-check=0 \
; RUN: -passes=polly-codegen -pass-remarks-analysis="polly-scops" \
; RUN: -disable-output < %s 2>&1 | FileCheck %s --check-prefix=NO_HEURISTIC

; void test(int **restrict a, int *restrict b, int *restrict c, int M) {
;   for (int k = 0; k < M; k++) // Vectorizable loop.
;     b[k] += c[k];
;
;   // Non-loop blocks.
;   // Load of a[0] is not relevant to vectorizable loop.
;   if (a[0][4])
;     a[0][4] = 0;
; }

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"

define void @test(ptr noalias nocapture noundef readonly %a, ptr noalias nocapture noundef %b, ptr noalias nocapture noundef readonly %c, i32 noundef %M) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %cmp11 = icmp sgt i32 %M, 0
  br i1 %cmp11, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry.split
  %wide.trip.count = zext nneg i32 %M to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %c, i64 %indvars.iv
  %i1 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %indvars.iv
  %i2 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %i2, %i1
  store i32 %add, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry.split
  %i3 = load ptr, ptr %a, align 8
  %arrayidx4 = getelementptr inbounds i8, ptr %i3, i64 16
  %i4 = load i32, ptr %arrayidx4, align 4
  %tobool.not = icmp eq i32 %i4, 0
  br i1 %tobool.not, label %if.end, label %if.then

if.then:                                          ; preds = %for.cond.cleanup
  store i32 0, ptr %arrayidx4, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.cond.cleanup
  ret void
}

; Check that region %entry.split---%for.cond.cleanup without trailing blocks
; %for.cond.cleanup---%if.end is detected with region expansion profitability check.
; HEURISTIC-LABLE: isl ast :: test :: %entry.split---%for.cond.cleanup
; HEURISTIC:      for (int c0 = 0; c0 < M; c0 += 1)
; HEURISTIC-NEXT:	Stmt_for_body(c0);

; Check that without region expansion heuristic, vectorization is missed.
; NO_HEURISTIC: SCoP begins here.
; NO_HEURISTIC-NEXT: Invariant load assumption:       [M] -> {  : false }
; NO_HEURISTIC-NEXT: SCoP ends here but was dismissed.
