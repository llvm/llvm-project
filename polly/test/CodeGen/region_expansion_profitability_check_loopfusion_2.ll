; RUN: opt -polly-use-llvm-names -polly-region-expansion-profitability-check=1 \
; RUN: -polly-loopfusion-greedy=1 "-passes=scop(polly-opt-isl,print<polly-ast>)" \
; RUN: -disable-output < %s | FileCheck %s

; void foo(int *A, int *B, int *C, int n, int *U) {
;   for (int i = 0; i < 1024; ++i) // L1
;     B[i] += A[i];
;
;   if (*U != 1) // Blocks with unrelated memory accesses.
;     *U+=42;
;
;   for (int j = 0; j < 1024; ++j) // L2
;     C[j] += B[j];
; }

define void @foo(ptr nocapture noundef readonly %A, ptr nocapture noundef %B, ptr nocapture noundef %C, i32 noundef %n, ptr nocapture noundef %U) {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  %0 = load i32, ptr %U, align 4
  %cmp3.not = icmp eq i32 %0, 1
  br i1 %cmp3.not, label %if.end, label %if.then

for.body:                                         ; preds = %entry.split, %for.body
  %indvars.iv = phi i64 [ 0, %entry.split ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %1 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %2 = load i32, ptr %arrayidx2, align 4
  %add = add nsw i32 %2, %1
  store i32 %add, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

if.then:                                          ; preds = %for.cond.cleanup
  %add4 = add nsw i32 %0, 42
  store i32 %add4, ptr %U, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.cond.cleanup
  br label %for.body8

for.cond.cleanup7:                                ; preds = %for.body8
  ret void

for.body8:                                        ; preds = %if.end, %for.body8
  %indvars.iv28 = phi i64 [ 0, %if.end ], [ %indvars.iv.next29, %for.body8 ]
  %arrayidx10 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv28
  %3 = load i32, ptr %arrayidx10, align 4
  %arrayidx12 = getelementptr inbounds i32, ptr %C, i64 %indvars.iv28
  %4 = load i32, ptr %arrayidx12, align 4
  %add13 = add nsw i32 %4, %3
  store i32 %add13, ptr %arrayidx12, align 4
  %indvars.iv.next29 = add nuw nsw i64 %indvars.iv28, 1
  %exitcond31.not = icmp eq i64 %indvars.iv.next29, 1024
  br i1 %exitcond31.not, label %for.cond.cleanup7, label %for.body8
}

; Check that if blocks with unrelated memory accesses are followed by another
; loop from region expansion, loop fusion is encouraged with region expansion
; profitability check. L1 and L2 are fused.
; CHECK-LABEL: isl ast :: foo :: %for.body---%for.cond.cleanup7
; CHECK:           for (int c0 = 0; c0 <= 1023; c0 += 1) {
; CHECK-NEXT:        Stmt_for_body(c0);
; CHECK-NEXT:        Stmt_for_body8(c0);
