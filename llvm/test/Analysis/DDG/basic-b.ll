; RUN: opt < %s -disable-output "-passes=print<ddg>" 2>&1 | FileCheck %s

; CHECK-LABEL: 'DDG' for loop 'test1.for.body':

; CHECK: Node Address:[[N9:0x[0-9a-f]*]]:pi-block
; CHECK-NEXT:--- start of nodes in pi-block ---
; CHECK: Node Address:[[N13:0x[0-9a-f]*]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %inc = add i64 %i.02, 1
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N14:0x[0-9a-f]*]]

; CHECK: Node Address:[[N14]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %i.02 = phi i64 [ %inc, %test1.for.body ], [ 1, %test1.for.body.preheader ]
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N13]]
; CHECK-NEXT:--- end of nodes in pi-block ---
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N1:0x[0-9a-f]*]]
; CHECK-NEXT:  [def-use] to [[N4:0x[0-9a-f]*]]
; CHECK-NEXT:  [def-use] to [[N6:0x[0-9a-f]*]]
; CHECK-NEXT:  [def-use] to [[N7:0x[0-9a-f]*]]

; CHECK: Node Address:[[N7]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %cmp = icmp ult i64 %inc, %sub
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N8:0x[0-9a-f]*]]

; CHECK: Node Address:[[N8]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    br i1 %cmp, label %test1.for.body, label %for.end.loopexit
; CHECK-NEXT: Edges:none!

; CHECK: Node Address:[[N6]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %arrayidx3 = getelementptr inbounds float, float* %a, i64 %i.02
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N3:0x[0-9a-f]*]]

; CHECK: Node Address:[[N4]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %sub1 = add i64 %i.02, -1
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N5:0x[0-9a-f]*]]

; CHECK: Node Address:[[N5]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %arrayidx2 = getelementptr inbounds float, float* %a, i64 %sub1
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N3]]

; CHECK: Node Address:[[N1]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %arrayidx = getelementptr inbounds float, float* %b, i64 %i.02
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N2:0x[0-9a-f]*]]

; CHECK: Node Address:[[N2]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %0 = load float, float* %arrayidx, align 4
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N3]]

; CHECK: Node Address:[[N3]]:pi-block
; CHECK-NEXT: --- start of nodes in pi-block ---
; CHECK: Node Address:[[N10:0x[0-9a-f]*]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %add = fadd float %0, %1
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N11:0x[0-9a-f]*]]

; CHECK: Node Address:[[N12:0x[0-9a-f]*]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %1 = load float, float* %arrayidx2, align 4
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N10]]

; CHECK: Node Address:[[N11]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    store float %add, float* %arrayidx3, align 4
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [memory] to [[N12]]
; CHECK-NEXT:--- end of nodes in pi-block ---
; CHECK-NEXT: Edges:none!



;; Loop-carried dependence requiring edge-reversal to expose a cycle
;; in the graph.
;; void test(unsigned long n, float * restrict a, float * restrict b) {
;;  for (unsigned long i = 1; i < n-1; i++)
;;    a[i] = b[i] + a[i-1];
;; }

define void @test1(i64 %n, float* noalias %a, float* noalias %b) {
entry:
  %sub = add i64 %n, -1
  %cmp1 = icmp ult i64 1, %sub
  br i1 %cmp1, label %test1.for.body, label %for.end

test1.for.body:                                         ; preds = %entry, %test1.for.body
  %i.02 = phi i64 [ %inc, %test1.for.body ], [ 1, %entry ]
  %arrayidx = getelementptr inbounds float, float* %b, i64 %i.02
  %0 = load float, float* %arrayidx, align 4
  %sub1 = add i64 %i.02, -1
  %arrayidx2 = getelementptr inbounds float, float* %a, i64 %sub1
  %1 = load float, float* %arrayidx2, align 4
  %add = fadd float %0, %1
  %arrayidx3 = getelementptr inbounds float, float* %a, i64 %i.02
  store float %add, float* %arrayidx3, align 4
  %inc = add i64 %i.02, 1
  %cmp = icmp ult i64 %inc, %sub
  br i1 %cmp, label %test1.for.body, label %for.end

for.end:                                          ; preds = %test1.for.body, %entry
  ret void
}

; CHECK-LABEL: 'DDG' for loop 'test2.for.body':

; CHECK: Node Address:[[N11:0x[0-9a-f]*]]:pi-block
; CHECK-NEXT:--- start of nodes in pi-block ---
; CHECK: Node Address:[[N12:0x[0-9a-f]*]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %inc = add i64 %i.02, 1
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N13:0x[0-9a-f]*]]

; CHECK: Node Address:[[N13]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %i.02 = phi i64 [ %inc, %test2.for.body ], [ 1, %test2.for.body.preheader ]
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N12]]
; CHECK-NEXT:--- end of nodes in pi-block ---
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N1:0x[0-9a-f]*]]
; CHECK-NEXT:  [def-use] to [[N4:0x[0-9a-f]*]]
; CHECK-NEXT:  [def-use] to [[N8:0x[0-9a-f]*]]
; CHECK-NEXT:  [def-use] to [[N9:0x[0-9a-f]*]]

; CHECK: Node Address:[[N9]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %cmp = icmp ult i64 %inc, %sub
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N10:0x[0-9a-f]*]]

; CHECK: Node Address:[[N10]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    br i1 %cmp, label %test2.for.body, label %for.end.loopexit
; CHECK-NEXT: Edges:none!

; CHECK: Node Address:[[N8]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %arrayidx3 = getelementptr inbounds float, float* %a, i64 %i.02
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N7:0x[0-9a-f]*]]

; CHECK: Node Address:[[N4]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %add1 = add i64 %i.02, 1
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N5:0x[0-9a-f]*]]

; CHECK: Node Address:[[N5]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %arrayidx2 = getelementptr inbounds float, float* %a, i64 %add1
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N6:0x[0-9a-f]*]]

; CHECK: Node Address:[[N6]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %1 = load float, float* %arrayidx2, align 4
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N3:0x[0-9a-f]*]]
; CHECK-NEXT:  [memory] to [[N7]]

; CHECK: Node Address:[[N1]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %arrayidx = getelementptr inbounds float, float* %b, i64 %i.02
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N2:0x[0-9a-f]*]]

; CHECK: Node Address:[[N2]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %0 = load float, float* %arrayidx, align 4
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N3]]


; CHECK: Node Address:[[N3]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %add = fadd float %0, %1
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N7]]

; CHECK: Node Address:[[N7]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    store float %add, float* %arrayidx3, align 4
; CHECK-NEXT: Edges:none!


;; Forward loop-carried dependence *not* causing a cycle.
;; void test2(unsigned long n, float * restrict a, float * restrict b) {
;;  for (unsigned long i = 1; i < n-1; i++)
;;    a[i] = b[i] + a[i+1];
;; }

define void @test2(i64 %n, float* noalias %a, float* noalias %b) {
entry:
  %sub = add i64 %n, -1
  %cmp1 = icmp ult i64 1, %sub
  br i1 %cmp1, label %test2.for.body, label %for.end

test2.for.body:                                         ; preds = %entry, %test2.for.body
  %i.02 = phi i64 [ %inc, %test2.for.body ], [ 1, %entry ]
  %arrayidx = getelementptr inbounds float, float* %b, i64 %i.02
  %0 = load float, float* %arrayidx, align 4
  %add1 = add i64 %i.02, 1
  %arrayidx2 = getelementptr inbounds float, float* %a, i64 %add1
  %1 = load float, float* %arrayidx2, align 4
  %add = fadd float %0, %1
  %arrayidx3 = getelementptr inbounds float, float* %a, i64 %i.02
  store float %add, float* %arrayidx3, align 4
  %inc = add i64 %i.02, 1
  %cmp = icmp ult i64 %inc, %sub
  br i1 %cmp, label %test2.for.body, label %for.end

for.end:                                          ; preds = %test2.for.body, %entry
  ret void
}
