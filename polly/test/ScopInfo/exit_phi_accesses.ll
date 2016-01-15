; RUN: opt %loadPolly -analyze -polly-scops %s | FileCheck %s

; Check that PHI nodes only create PHI access and nothing else (e.g. unnecessary
; SCALAR accesses). In this case, for a PHI in the exit node, hence there is no
; PHI ReadAccess.

; CHECK-LABEL: Function: foo
;
; CHECK:       Statements {
; CHECK-NEXT:      Stmt_header
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              { Stmt_header[] };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              { Stmt_header[] -> [0, 0] };
; CHECK-NEXT:          MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              { Stmt_header[] -> MemRef_phi[] };
; CHECK-NEXT:      Stmt_body
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              { Stmt_body[i0] : 0 <= i0 <= 100 };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              { Stmt_body[i0] -> [1, i0] };
; CHECK-NEXT:          MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              { Stmt_body[i0] -> MemRef_phi[] };
; CHECK-NEXT:  }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define float @foo(float %sum, float* %A) {
entry:
  br label %header

header:
  br i1 true, label %body, label %exit

body:
  %i = phi i64 [ 0, %header ], [ %next, %body ]
  %scalar = fadd float 0.0, 0.0
  %next = add nuw nsw i64 %i, 1
  %cond = icmp ne i64 %i, 100
  br i1 %cond, label %body, label %exit

exit:
  %phi = phi float [%scalar, %body], [0.0, %header]
  ret float %phi
}
