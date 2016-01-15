; RUN: opt %loadPolly -analyze -polly-scops %s | FileCheck %s

; CHECK-LABEL: Function: foo
;
; CHECK:       Statements {
; CHECK-NEXT:      Stmt_body
; CHECK-NEXT:          Domain :=
; CHECK-NEXT:              { Stmt_body[i0] : i0 <= 100 and i0 >= 0 };
; CHECK-NEXT:          Schedule :=
; CHECK-NEXT:              { Stmt_body[i0] -> [i0] };
; CHECK-NEXT:          MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              { Stmt_body[i0] -> MemRef_sum__phi[] };
; CHECK-NEXT:          ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              { Stmt_body[i0] -> MemRef_sum__phi[] };
; CHECK-NEXT:          ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:              { Stmt_body[i0] -> MemRef_A[i0] };
; CHECK-NEXT:          MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:              { Stmt_body[i0] -> MemRef_sum_next[] };
; CHECK-NEXT:  }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define float @foo(float* %A) {
entry:
  br label %header

header:
  fence seq_cst
  br i1 true, label %body, label %exit

body:
  %i = phi i64 [ 0, %header ], [ %next, %body ]
  %sum = phi float [ 0.0, %header ], [ %sum.next, %body ]
  %arrayidx = getelementptr float, float* %A, i64 %i
  %scalar = fadd float 0.0, 0.0
  %next = add nuw nsw i64 %i, 1
  %val = load float, float* %arrayidx
  %sum.next = fadd float %sum, %val
  %cond = icmp ne i64 %i, 100
  br i1 %cond, label %body, label %after

after:
  br label %exit

exit:
  %phi = phi float [%sum.next, %after], [0.0, %header]
  ret float %phi
}
