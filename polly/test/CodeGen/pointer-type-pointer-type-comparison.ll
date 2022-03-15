; RUN: opt %loadPolly -polly-print-ast -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s -check-prefix=CODEGEN
;

;    void f(int a[], int N, float *P, float *Q) {
;      int i;
;      for (i = 0; i < N; ++i)
;        if (P != Q)
;          a[i] = i;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i64* nocapture %a, i64 %N, float * %P, float * %Q) nounwind {
entry:
  br label %bb

bb:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %bb.backedge ]
  %brcond = icmp ne float* %P, %Q
  br i1 %brcond, label %store, label %bb.backedge

store:
  %scevgep = getelementptr inbounds i64, i64* %a, i64 %i
  store i64 %i, i64* %scevgep
  br label %bb.backedge

bb.backedge:
  %i.inc = add nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.inc, %N
  br i1 %exitcond, label %return, label %bb

return:
  ret void
}

; CHECK:      if (Q >= P + 1 || P >= Q + 1)
; CHECK-NEXT:   for (int c0 = 0; c0 < N; c0 += 1)
; CHECK-NEXT:     Stmt_store(c0);

; CODEGEN:       polly.cond:
; CODEGEN-NEXT:  %[[Q:[_a-zA-Z0-9]+]] = ptrtoint float* %Q to i64
; CODEGEN-NEXT:  %[[P:[_a-zA-Z0-9]+]] = ptrtoint float* %P to i64
; CODEGEN-NEXT:  %[[PInc:[_a-zA-Z0-9]+]] = add nsw i64 %[[P]], 1
; CODEGEN-NEXT:  %[[CMP:[_a-zA-Z0-9]+]] = icmp sge i64 %[[Q]], %[[PInc]]
; CODEGEN-NEXT:  %[[P2:[_a-zA-Z0-9]+]] = ptrtoint float* %P to i64
; CODEGEN-NEXT:  %[[Q2:[_a-zA-Z0-9]+]] = ptrtoint float* %Q to i64
; CODEGEN-NEXT:  %[[QInc:[_a-zA-Z0-9]+]] = add nsw i64 %[[Q2]], 1
; CODEGEN-NEXT:  %[[CMP2:[_a-zA-Z0-9]+]] = icmp sge i64 %[[P2]], %[[QInc]]
; CODEGEN-NEXT:  %[[CMP3:[_a-zA-Z0-9]+]] = or i1 %[[CMP]], %[[CMP2]]
; CODEGEN-NEXT:  br i1 %[[CMP3]]
