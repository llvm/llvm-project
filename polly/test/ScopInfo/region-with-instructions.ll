; RUN: opt %loadPolly -polly-print-scops -polly-print-instructions -disable-output < %s | FileCheck %s

; CHECK: Statements {
; CHECK: 	Stmt_bb46
; CHECK:         Domain :=
; CHECK:             [tmp44, tmp9] -> { Stmt_bb46[] : tmp9 = tmp44 };
; CHECK:         Schedule :=
; CHECK:             [tmp44, tmp9] -> { Stmt_bb46[] -> [0, 0] };
; CHECK:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK:             [tmp44, tmp9] -> { Stmt_bb46[] -> MemRef_tmp47[] };
; CHECK:         Instructions {
; CHECK:               %tmp47 = or i64 1, %tmp14
; CHECK:         }
; CHECK: 	Stmt_bb48__TO__bb56
; CHECK:         Domain :=
; CHECK:             [tmp44, tmp9] -> { Stmt_bb48__TO__bb56[i0] : tmp9 = tmp44 and 0 <= i0 < tmp44 };
; CHECK:         Schedule :=
; CHECK:             [tmp44, tmp9] -> { Stmt_bb48__TO__bb56[i0] -> [1, i0] };
; CHECK:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK:             [tmp44, tmp9] -> { Stmt_bb48__TO__bb56[i0] -> MemRef_A[i0] };
; CHECK:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK:             [tmp44, tmp9] -> { Stmt_bb48__TO__bb56[i0] -> MemRef_A[i0] };
; CHECK:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK:             [tmp44, tmp9] -> { Stmt_bb48__TO__bb56[i0] -> MemRef_tmp47[] };
; CHECK:         MayWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK:             [tmp44, tmp9] -> { Stmt_bb48__TO__bb56[i0] -> MemRef_A[i0] };
; CHECK:         Instructions {
; CHECK:               %tmp51 = load i64, i64* %tmp50, align 8
; CHECK:               %tmp52 = and i64 %tmp51, %tmp26
; CHECK:               %tmp53 = icmp eq i64 %tmp52, %tmp26
; CHECK:               store i64 42, i64* %tmp50, align 8
; CHECK:         }
; CHECK: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @quux(i32 %arg, i32 %arg1, i64* %A, i64 %tmp9, i64 %tmp24, i64 %tmp14, i64 %tmp22, i64 %tmp44) {
bb:
  %tmp26 = or i64 %tmp22, %tmp24
  br label %bb39

bb39:                                             ; preds = %bb39, %bb38
  %tmp45 = icmp eq i64 %tmp44, %tmp9
  br i1 %tmp45, label %bb46, label %bb81

bb46:                                             ; preds = %bb39
  %tmp47 = or i64 1, %tmp14
  br label %bb48

bb48:                                             ; preds = %bb56, %bb46
  %tmp49 = phi i64 [ 0, %bb46 ], [ %tmp57, %bb56 ]
  %tmp50 = getelementptr inbounds i64, i64* %A, i64 %tmp49
  %tmp51 = load i64, i64* %tmp50, align 8
  %tmp52 = and i64 %tmp51, %tmp26
  %tmp53 = icmp eq i64 %tmp52, %tmp26
  store i64 42, i64* %tmp50, align 8
  br i1 %tmp53, label %bb54, label %bb56

bb54:                                             ; preds = %bb48
  %tmp55 = xor i64 %tmp51, %tmp47
  store i64 %tmp55, i64* %tmp50, align 8
  br label %bb56

bb56:                                             ; preds = %bb54, %bb48
  %tmp57 = add nuw nsw i64 %tmp49, 1
  %tmp58 = icmp eq i64 %tmp57, %tmp9
  br i1 %tmp58, label %bb81, label %bb48

bb81:                                             ; preds = %bb74, %bb56
  ret void
}
