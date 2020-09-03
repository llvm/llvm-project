; RUN: opt %loadPolly -basic-aa -polly-scops \
; RUN:     -polly-allow-nonaffine -polly-allow-nonaffine-branches \
; RUN:     -polly-allow-nonaffine-loops=true -analyze < %s | FileCheck %s \
; RUN:     -check-prefix=SCALAR
; RUN: opt %loadPolly -basic-aa -polly-scops -polly-allow-nonaffine \
; RUN:     -polly-process-unprofitable=false \
; RUN:     -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=true \
; RUN:     -analyze < %s | FileCheck %s -check-prefix=PROFIT
;
; SCALAR:      Function: f
; SCALAR-NEXT: Region: %bb1---%bb13
; SCALAR-NEXT: Max Loop Depth:  1
; SCALAR-NEXT: Invariant Accesses: {
; SCALAR-NEXT: }
; SCALAR-NEXT: Context:
; SCALAR-NEXT: {  :  }
; SCALAR-NEXT: Assumed Context:
; SCALAR-NEXT: {  :  }
; SCALAR-NEXT: Invalid Context:
; SCALAR-NEXT: {  : false }
; SCALAR-NEXT: Arrays {
; SCALAR-NEXT:     i32 MemRef_C[*]; // Element size 4
; SCALAR-NEXT:     i32 MemRef_A[*]; // Element size 4
; SCALAR-NEXT: }
; SCALAR-NEXT: Arrays (Bounds as pw_affs) {
; SCALAR-NEXT:     i32 MemRef_C[*]; // Element size 4
; SCALAR-NEXT:     i32 MemRef_A[*]; // Element size 4
; SCALAR-NEXT: }
; SCALAR-NEXT: Alias Groups (0):
; SCALAR-NEXT:     n/a
; SCALAR-NEXT: Statements {
; SCALAR-NEXT:     Stmt_bb3__TO__bb11
; SCALAR-NEXT:         Domain :=
; SCALAR-NEXT:             { Stmt_bb3__TO__bb11[i0] : 0 <= i0 <= 1023 };
; SCALAR-NEXT:         Schedule :=
; SCALAR-NEXT:             { Stmt_bb3__TO__bb11[i0] -> [i0] };
; SCALAR-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; SCALAR-NEXT:             { Stmt_bb3__TO__bb11[i0] -> MemRef_C[i0] };
; SCALAR-NEXT:         ReadAccess :=    [Reduction Type: +] [Scalar: 0]
; SCALAR-NEXT:             { Stmt_bb3__TO__bb11[i0] -> MemRef_A[o0] };
; SCALAR-NEXT:         MayWriteAccess :=    [Reduction Type: +] [Scalar: 0]
; SCALAR-NEXT:             { Stmt_bb3__TO__bb11[i0] -> MemRef_A[o0] };
; SCALAR-NEXT: }

; PROFIT-NOT: Statements
;
;    void f(int * restrict A, int * restrict C) {
;      int j;
;      for (int i = 0; i < 1024; i++) {
;        while ((j = C[i++])) {
;          A[j]++;
;          if (true) break;
;        }
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* noalias %A, i32* noalias %C) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb12, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb12 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb13

bb2:                                              ; preds = %bb1
  br label %bb3

bb3:                                              ; preds = %bb6, %bb2
  %tmp = getelementptr inbounds i32, i32* %C, i64 %indvars.iv
  %tmp4 = load i32, i32* %tmp, align 4
  %tmp5 = icmp eq i32 %tmp4, 0
  br i1 %tmp5, label %bb11, label %bb6

bb6:                                              ; preds = %bb3
  %tmp7 = sext i32 %tmp4 to i64
  %tmp8 = getelementptr inbounds i32, i32* %A, i64 %tmp7
  %tmp9 = load i32, i32* %tmp8, align 4
  %tmp10 = add nsw i32 %tmp9, 1
  store i32 %tmp10, i32* %tmp8, align 4
  br i1 true, label %bb11, label %bb3

bb11:                                             ; preds = %bb3
  br label %bb12

bb12:                                             ; preds = %bb11
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb13:                                             ; preds = %bb1
  ret void
}
