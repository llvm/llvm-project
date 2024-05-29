; RUN: opt %loadNPMPolly '-passes=print<polly-ast>' -disable-output < %s | FileCheck %s
; RUN: opt %loadNPMPolly -passes=polly-codegen -S < %s | FileCheck %s -check-prefix=CODEGEN

; void f(int a[], int N, float *P) {
;   int i;
;   for (i = 0; i < N; ++i)
;     if (P != 0)
;       a[i] = i;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @f(ptr nocapture %a, i64 %N, ptr %P) nounwind {
entry:
  br label %bb

bb:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %bb.backedge ]
  %brcond = icmp ne ptr %P, null
  br i1 %brcond, label %store, label %bb.backedge

store:
  %scevgep = getelementptr i64, ptr %a, i64 %i
  store i64 %i, ptr %scevgep
  br label %bb.backedge

bb.backedge:
  %i.inc = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.inc, %N
  br i1 %exitcond, label %return, label %bb

return:
  ret void
}

; CHECK:      if (P <= -1 || P >= 1)
; CHECK-NEXT:   for (int c0 = 0; c0 < N; c0 += 1)
; CHECK-NEXT:     Stmt_store(c0);

; CODEGEN-LABEL: polly.cond:
; CODEGEN-NEXT:   %[[R1:[0-9]*]] = ptrtoint ptr %P to i64
; CODEGEN-NEXT:   %[[R2:[0-9]*]] = icmp sle i64 %[[R1]], -1
; CODEGEN-NEXT:   %[[R3:[0-9]*]] = ptrtoint ptr %P to i64
; CODEGEN-NEXT:   %[[R4:[0-9]*]] = icmp sge i64 %[[R3]], 1
; CODEGEN-NEXT:   %[[R5:[0-9]*]] = or i1 %[[R2]], %[[R4]]
; CODEGEN-NEXT:   br i1 %[[R5]]

