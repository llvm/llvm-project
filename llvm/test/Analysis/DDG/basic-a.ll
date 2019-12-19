; RUN: opt < %s -disable-output "-passes=print<ddg>" 2>&1 | FileCheck %s

; CHECK-LABEL: 'DDG' for loop 'test1.for.body':

; CHECK: Node Address:[[PI:0x[0-9a-f]*]]:pi-block
; CHECK-NEXT: --- start of nodes in pi-block ---
; CHECK: Node Address:[[N1:0x[0-9a-f]*]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %i.02 = phi i64 [ %inc, %test1.for.body ], [ 0, %test1.for.body.preheader ]
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N2:0x[0-9a-f]*]]

; CHECK: Node Address:[[N2]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %inc = add i64 %i.02, 1
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N1]]
; CHECK-NEXT: --- end of nodes in pi-block ---
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N3:0x[0-9a-f]*]]
; CHECK-NEXT:  [def-use] to [[N4:0x[0-9a-f]*]]
; CHECK-NEXT:  [def-use] to [[N5:0x[0-9a-f]*]]

; CHECK: Node Address:[[N5]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %exitcond = icmp ne i64 %inc, %n
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N6:0x[0-9a-f]*]]

; CHECK: Node Address:[[N6]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    br i1 %exitcond, label %test1.for.body, label %for.end.loopexit
; CHECK-NEXT: Edges:none!

; CHECK: Node Address:[[N4]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %arrayidx1 = getelementptr inbounds float, float* %a, i64 %i.02
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N7:0x[0-9a-f]*]]

; CHECK: Node Address:[[N3]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %arrayidx = getelementptr inbounds float, float* %b, i64 %i.02
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N8:0x[0-9a-f]*]]

; CHECK: Node Address:[[N8]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %0 = load float, float* %arrayidx, align 4
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N9:0x[0-9a-f]*]]

; CHECK: Node Address:[[N10:0x[0-9a-f]*]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %conv = uitofp i64 %n to float
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N9]]

; CHECK: Node Address:[[N9]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %add = fadd float %0, %conv
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N7]]

; CHECK: Node Address:[[N7]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    store float %add, float* %arrayidx1, align 4
; CHECK-NEXT: Edges:none!


;; No memory dependencies.
;; void test1(unsigned long n, float * restrict a, float * restrict b) {
;;  for (unsigned long i = 0; i < n; i++)
;;    a[i] = b[i] + n;
;; }

define void @test1(i64 %n, float* noalias %a, float* noalias %b) {
entry:
  %exitcond1 = icmp ne i64 0, %n
  br i1 %exitcond1, label %test1.for.body, label %for.end

test1.for.body:                                         ; preds = %entry, %test1.for.body
  %i.02 = phi i64 [ %inc, %test1.for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %b, i64 %i.02
  %0 = load float, float* %arrayidx, align 4
  %conv = uitofp i64 %n to float
  %add = fadd float %0, %conv
  %arrayidx1 = getelementptr inbounds float, float* %a, i64 %i.02
  store float %add, float* %arrayidx1, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, %n
  br i1 %exitcond, label %test1.for.body, label %for.end

for.end:                                          ; preds = %test1.for.body, %entry
  ret void
}


; CHECK-LABEL: 'DDG' for loop 'test2.for.body':

; CHECK: Node Address:[[PI:0x[0-9a-f]*]]:pi-block
; CHECK-NEXT: --- start of nodes in pi-block ---
; CHECK: Node Address:[[N1:0x[0-9a-f]*]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %i.02 = phi i64 [ %inc, %test2.for.body ], [ 0, %test2.for.body.preheader ]
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N2:0x[0-9a-f]*]]

; CHECK: Node Address:[[N2]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %inc = add i64 %i.02, 1
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N1]]
; CHECK-NEXT: --- end of nodes in pi-block ---
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N3:0x[0-9a-f]*]]
; CHECK-NEXT:  [def-use] to [[N4:0x[0-9a-f]*]]
; CHECK-NEXT:  [def-use] to [[N5:0x[0-9a-f]*]]
; CHECK-NEXT:  [def-use] to [[N6:0x[0-9a-f]*]]

; CHECK: Node Address:[[N6]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %exitcond = icmp ne i64 %inc, %n
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N7:0x[0-9a-f]*]]

; CHECK: Node Address:[[N7]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    br i1 %exitcond, label %test2.for.body, label %for.end.loopexit
; CHECK-NEXT: Edges:none!

; CHECK: Node Address:[[N5]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %arrayidx2 = getelementptr inbounds float, float* %a, i64 %i.02
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N8:0x[0-9a-f]*]]

; CHECK: Node Address:[[N4]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %arrayidx1 = getelementptr inbounds float, float* %a, i64 %i.02
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N9:0x[0-9a-f]*]]

; CHECK: Node Address:[[N9]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %1 = load float, float* %arrayidx1, align 4
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N20:0x[0-9a-f]*]]
; CHECK-NEXT:  [memory] to [[N8]]

; CHECK: Node Address:[[N3]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %arrayidx = getelementptr inbounds float, float* %b, i64 %i.02
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N10:0x[0-9a-f]*]]

; CHECK: Node Address:[[N10]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %0 = load float, float* %arrayidx, align 4
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N20]]

; CHECK: Node Address:[[N20]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    %add = fadd float %0, %1
; CHECK-NEXT: Edges:
; CHECK-NEXT:  [def-use] to [[N8]]

; CHECK: Node Address:[[N8]]:single-instruction
; CHECK-NEXT: Instructions:
; CHECK-NEXT:    store float %add, float* %arrayidx2, align 4
; CHECK-NEXT: Edges:none!



;; Loop-independent memory dependencies.
;; void test2(unsigned long n, float * restrict a, float * restrict b) {
;;  for (unsigned long i = 0; i < n; i++)
;;    a[i] = b[i] + a[i];
;; }

define void @test2(i64 %n, float* noalias %a, float* noalias %b) {
entry:
  %exitcond1 = icmp ne i64 0, %n
  br i1 %exitcond1, label %test2.for.body, label %for.end

test2.for.body:                                         ; preds = %entry, %test2.for.body
  %i.02 = phi i64 [ %inc, %test2.for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %b, i64 %i.02
  %0 = load float, float* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, float* %a, i64 %i.02
  %1 = load float, float* %arrayidx1, align 4
  %add = fadd float %0, %1
  %arrayidx2 = getelementptr inbounds float, float* %a, i64 %i.02
  store float %add, float* %arrayidx2, align 4
  %inc = add i64 %i.02, 1
  %exitcond = icmp ne i64 %inc, %n
  br i1 %exitcond, label %test2.for.body, label %for.end

for.end:                                          ; preds = %test2.for.body, %entry
  ret void
}