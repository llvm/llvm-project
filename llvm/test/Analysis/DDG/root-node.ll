; RUN: opt < %s -disable-output "-passes=print<ddg>" 2>&1 | FileCheck %s

; CHECK-LABEL: 'DDG' for loop 'test1.for.body':

; CHECK: Node Address:[[ROOT:0x[0-9a-f]*]]:root
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [rooted] to [[N1:0x[0-9a-f]*]]
; CHECK-NEXT:  [rooted] to [[N2:0x[0-9a-f]*]]

; CHECK: Node Address:[[N2]]:pi-block
; CHECK: %i1.02 = phi i64 [ 0, %for.body.lr.ph ], [ %inc, %test1.for.body ]

; CHECK: Node Address:[[N1]]:pi-block
; CHECK: %i2.03 = phi i64 [ 0, %for.body.lr.ph ], [ %inc2, %test1.for.body ]

;; // Two separate components in the graph. Root node must link to both.
;; void test1(unsigned long n, float * restrict a, float * restrict b) {
;;   for (unsigned long i1 = 0, i2 = 0; i1 < n; i1++, i2++) {
;;     a[i1] = 1;
;;     b[i2] = -1;
;;   }
;; }

define void @test1(i64 %n, float* noalias %a, float* noalias %b) {
entry:
  %cmp1 = icmp ult i64 0, %n
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %test1.for.body

test1.for.body:                                         ; preds = %for.body.lr.ph, %test1.for.body
  %i2.03 = phi i64 [ 0, %for.body.lr.ph ], [ %inc2, %test1.for.body ]
  %i1.02 = phi i64 [ 0, %for.body.lr.ph ], [ %inc, %test1.for.body ]
  %arrayidx = getelementptr inbounds float, float* %a, i64 %i1.02
  store float 1.000000e+00, float* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, float* %b, i64 %i2.03
  store float -1.000000e+00, float* %arrayidx1, align 4
  %inc = add i64 %i1.02, 1
  %inc2 = add i64 %i2.03, 1
  %cmp = icmp ult i64 %inc, %n
  br i1 %cmp, label %test1.for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %test1.for.body
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void
}
