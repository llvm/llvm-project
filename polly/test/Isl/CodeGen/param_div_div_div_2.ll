; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s --check-prefix=IR
;
; Check that we guard the divisions because we moved them and thereby increased
; their domain.
;
; CHECK:         Invalid Context:
; CHECK-NEXT:    [p_0] -> {  : p_0 < 0 }
; CHECK-NEXT:    p0: ((%a /u %b) /u (%c /u %d))
;
;    void f(unsigned *A, unsigned a, unsigned b, unsigned c, unsigned d) {
;      for (unsigned i; i < 100; i++)
;        A[i] += A[(a / b) / (c / d)];
;    }
;
; IR:       %[[R0:[.a-zA-Z0-9]*]] = icmp ugt i32 %b, 1
; IR-NEXT:  %[[R1:[.a-zA-Z0-9]*]] = select i1 %[[R0]], i32 %b, i32 1
; IR-NEXT:  %[[R2:[.a-zA-Z0-9]*]] = udiv i32 %a, %[[R1]]
; IR-NEXT:  %[[R5:[.a-zA-Z0-9]*]] = icmp ugt i32 %d, 1
; IR-NEXT:  %[[R6:[.a-zA-Z0-9]*]] = select i1 %[[R5]], i32 %d, i32 1
; IR-NEXT:  %[[R7:[.a-zA-Z0-9]*]] = udiv i32 %c, %[[R6]]
; IR-NEXT:  %[[R3:[.a-zA-Z0-9]*]] = icmp ugt i32 %[[R7]], 1
; IR-NEXT:  %[[R4:[.a-zA-Z0-9]*]] = select i1 %[[R3]], i32 %[[R7]], i32 1
; IR-NEXT:  %[[R8:[.a-zA-Z0-9]*]] = udiv i32 %[[R2]], %[[R4]]
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp ult i64 %indvars.iv, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %div = udiv i32 %a, %b
  %div1 = udiv i32 %c, %d
  %div2 = udiv i32 %div, %div1
  %idxprom = zext i32 %div2 to i64
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom
  %tmp = load i32, i32* %arrayidx, align 4
  %arrayidx4 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp1 = load i32, i32* %arrayidx4, align 4
  %add = add i32 %tmp1, %tmp
  store i32 %add, i32* %arrayidx4, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
