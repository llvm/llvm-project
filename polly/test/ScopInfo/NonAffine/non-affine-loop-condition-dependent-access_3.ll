; RUN: opt %loadPolly -basicaa -polly-scops -polly-allow-nonaffine-branches \
; RUN:     -polly-allow-nonaffine-loops=false \
; RUN:     -analyze < %s | FileCheck %s --check-prefix=INNERMOST
; RUN: opt %loadPolly -basicaa -polly-scops -polly-allow-nonaffine-branches \
; RUN:     -polly-allow-nonaffine-loops=true \
; RUN:      -analyze < %s | FileCheck %s --check-prefix=INNERMOST
; RUN: opt %loadPolly -basicaa -polly-scops -polly-allow-nonaffine \
; RUN:     -polly-allow-nonaffine-branches -polly-allow-nonaffine-loops=true \
; RUN:     -analyze < %s | FileCheck %s --check-prefix=ALL
;
; Here we have a non-affine loop (in the context of the loop nest)
; and also a non-affine access (A[k]). While we can always model the
; innermost loop as a SCoP of depth 1, we can overapproximate the
; innermost loop in the whole loop nest and model A[k] as a non-affine
; access.
;
; INNERMOST:      Function: f
; INNERMOST-NEXT: Region: %bb15---%bb13
; INNERMOST-NEXT: Max Loop Depth:  1
; INNERMOST-NEXT: Invariant Accesses: {
; INNERMOST-NEXT: }
; INNERMOST-NEXT: Context:
; INNERMOST-NEXT: [p_0, p_1, p_2] -> {  : 0 <= p_0 <= 2147483647 and 0 <= p_1 <= 4096 and 0 <= p_2 <= 4096 }
; INNERMOST-NEXT: Assumed Context:
; INNERMOST-NEXT: [p_0, p_1, p_2] -> {  :  }
; INNERMOST-NEXT: Boundary Context:
; INNERMOST-NEXT: [p_0, p_1, p_2] -> {  :  }
; INNERMOST-NEXT: p0: {0,+,{0,+,1}<nuw><nsw><%bb11>}<nuw><nsw><%bb13>
; INNERMOST-NEXT: p1: {0,+,4}<nuw><nsw><%bb11>
; INNERMOST-NEXT: p2: {0,+,4}<nuw><nsw><%bb13>
; INNERMOST-NEXT: Arrays {
; INNERMOST-NEXT:     i32 MemRef_A[*]; // Element size 4
; INNERMOST-NEXT:     i64 MemRef_indvars_iv_next6; // Element size 8
; INNERMOST-NEXT:     i32 MemRef_indvars_iv_next4; // Element size 4
; INNERMOST-NEXT: }
; INNERMOST-NEXT: Arrays (Bounds as pw_affs) {
; INNERMOST-NEXT:     i32 MemRef_A[*]; // Element size 4
; INNERMOST-NEXT:     i64 MemRef_indvars_iv_next6; // Element size 8
; INNERMOST-NEXT:     i32 MemRef_indvars_iv_next4; // Element size 4
; INNERMOST-NEXT: }
; INNERMOST-NEXT: Alias Groups (0):
; INNERMOST-NEXT:     n/a
; INNERMOST-NEXT: Statements {
; INNERMOST-NEXT:     Stmt_bb16
; INNERMOST-NEXT:         Domain :=
; INNERMOST-NEXT:             [p_0, p_1, p_2] -> { Stmt_bb16[i0] : 0 <= i0 < p_0 };
; INNERMOST-NEXT:         Schedule :=
; INNERMOST-NEXT:             [p_0, p_1, p_2] -> { Stmt_bb16[i0] -> [0, i0] };
; INNERMOST-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; INNERMOST-NEXT:             [p_0, p_1, p_2] -> { Stmt_bb16[i0] -> MemRef_A[o0] : 4o0 = p_1 };
; INNERMOST-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; INNERMOST-NEXT:             [p_0, p_1, p_2] -> { Stmt_bb16[i0] -> MemRef_A[o0] : 4o0 = p_2 };
; INNERMOST-NEXT:         ReadAccess :=    [Reduction Type: +] [Scalar: 0]
; INNERMOST-NEXT:             [p_0, p_1, p_2] -> { Stmt_bb16[i0] -> MemRef_A[i0] };
; INNERMOST-NEXT:         MustWriteAccess :=    [Reduction Type: +] [Scalar: 0]
; INNERMOST-NEXT:             [p_0, p_1, p_2] -> { Stmt_bb16[i0] -> MemRef_A[i0] };
; INNERMOST-NEXT:     Stmt_bb26
; INNERMOST-NEXT:         Domain :=
; INNERMOST-NEXT:             [p_0, p_1, p_2] -> { Stmt_bb26[] : p_0 >= 0 };
; INNERMOST-NEXT:         Schedule :=
; INNERMOST-NEXT:             [p_0, p_1, p_2] -> { Stmt_bb26[] -> [1, 0] };
; INNERMOST-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; INNERMOST-NEXT:             [p_0, p_1, p_2] -> { Stmt_bb26[] -> MemRef_indvars_iv_next6[] };
; INNERMOST-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; INNERMOST-NEXT:             [p_0, p_1, p_2] -> { Stmt_bb26[] -> MemRef_indvars_iv_next4[] };
; INNERMOST-NEXT: }

; ALL:      Function: f
; ALL-NEXT: Region: %bb11---%bb29
; ALL-NEXT: Max Loop Depth:  2
; ALL-NEXT: Invariant Accesses: {
; ALL-NEXT: }
; ALL-NEXT: Context:
; ALL-NEXT: {  :  }
; ALL-NEXT: Assumed Context:
; ALL-NEXT: {  :  }
; ALL-NEXT: Boundary Context:
; ALL-NEXT: {  :  }
; ALL-NEXT: Arrays {
; ALL-NEXT:     i32 MemRef_A[*]; // Element size 4
; ALL-NEXT: }
; ALL-NEXT: Arrays (Bounds as pw_affs) {
; ALL-NEXT:     i32 MemRef_A[*]; // Element size 4
; ALL-NEXT: }
; ALL-NEXT: Alias Groups (0):
; ALL-NEXT:     n/a
; ALL-NEXT: Statements {
; ALL-NEXT:     Stmt_bb15__TO__bb25
; ALL-NEXT:         Domain :=
; ALL-NEXT:             { Stmt_bb15__TO__bb25[i0, i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1023 };
; ALL-NEXT:         Schedule :=
; ALL-NEXT:             { Stmt_bb15__TO__bb25[i0, i1] -> [i0, i1] };
; ALL-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; ALL-NEXT:             { Stmt_bb15__TO__bb25[i0, i1] -> MemRef_A[i0] };
; ALL-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; ALL-NEXT:             { Stmt_bb15__TO__bb25[i0, i1] -> MemRef_A[i1] };
; ALL-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; ALL-NEXT:             { Stmt_bb15__TO__bb25[i0, i1] -> MemRef_A[o0] : 0 <= o0 <= 4294967295 };
; ALL-NEXT:         MayWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; ALL-NEXT:             { Stmt_bb15__TO__bb25[i0, i1] -> MemRef_A[o0] : 0 <= o0 <= 4294967295 };
; ALL-NEXT: }
;
;    void f(int *A) {
;      for (int i = 0; i < 1024; i++)
;        for (int j = 0; j < 1024; j++)
;          for (int k = 0; k < i * j; k++)
;            A[k] += A[i] + A[j];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A) {
bb:
  br label %bb11

bb11:                                             ; preds = %bb28, %bb
  %indvars.iv8 = phi i64 [ %indvars.iv.next9, %bb28 ], [ 0, %bb ]
  %indvars.iv1 = phi i32 [ %indvars.iv.next2, %bb28 ], [ 0, %bb ]
  %exitcond10 = icmp ne i64 %indvars.iv8, 1024
  br i1 %exitcond10, label %bb12, label %bb29

bb12:                                             ; preds = %bb11
  br label %bb13

bb13:                                             ; preds = %bb26, %bb12
  %indvars.iv5 = phi i64 [ %indvars.iv.next6, %bb26 ], [ 0, %bb12 ]
  %indvars.iv3 = phi i32 [ %indvars.iv.next4, %bb26 ], [ 0, %bb12 ]
  %exitcond7 = icmp ne i64 %indvars.iv5, 1024
  br i1 %exitcond7, label %bb14, label %bb27

bb14:                                             ; preds = %bb13
  br label %bb15

bb15:                                             ; preds = %bb24, %bb14
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb24 ], [ 0, %bb14 ]
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp ne i32 %lftr.wideiv, %indvars.iv3
  br i1 %exitcond, label %bb16, label %bb25

bb16:                                             ; preds = %bb15
  %tmp = getelementptr inbounds i32, i32* %A, i64 %indvars.iv8
  %tmp17 = load i32, i32* %tmp, align 4
  %tmp18 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv5
  %tmp19 = load i32, i32* %tmp18, align 4
  %tmp20 = add nsw i32 %tmp17, %tmp19
  %tmp21 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp22 = load i32, i32* %tmp21, align 4
  %tmp23 = add nsw i32 %tmp22, %tmp20
  store i32 %tmp23, i32* %tmp21, align 4
  br label %bb24

bb24:                                             ; preds = %bb16
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb15

bb25:                                             ; preds = %bb15
  br label %bb26

bb26:                                             ; preds = %bb25
  %indvars.iv.next6 = add nuw nsw i64 %indvars.iv5, 1
  %indvars.iv.next4 = add nuw nsw i32 %indvars.iv3, %indvars.iv1
  br label %bb13

bb27:                                             ; preds = %bb13
  br label %bb28

bb28:                                             ; preds = %bb27
  %indvars.iv.next9 = add nuw nsw i64 %indvars.iv8, 1
  %indvars.iv.next2 = add nuw nsw i32 %indvars.iv1, 1
  br label %bb11

bb29:                                             ; preds = %bb11
  ret void
}
