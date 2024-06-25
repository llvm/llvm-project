; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; The load access to A has a pointer-bitcast to another elements size before the
; GetElementPtr. Verify that we do not the GEP delinearization because it
; mismatches with the size of the loaded element type.
;
;    void f(short A[][4], int N, int P) {
;      short(*B)[4] = &A[P][0];
;      for (int i = 0; i < N; i++)
;        *((<4 x short> *)&A[7 * i][0]) = *((<4 x short>)&B[7 * i][0]);
;    }
;
define void @f(ptr %A, i32 %N, i32 %P) {
entry:
  %arrayidx1 = getelementptr inbounds [4 x i16], ptr %A, i32 %P, i64 0
  br label %for.cond

for.cond:
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i32 %indvars.iv, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %mul = mul nsw i32 %indvars.iv, 7
  %arrayidx4 = getelementptr inbounds [4 x i16], ptr %arrayidx1, i32 %mul, i64 0
  %tmp3 = load <4 x i16>, ptr %arrayidx4
  %arrayidx8 = getelementptr inbounds [4 x i16], ptr %A, i32 %mul, i64 0
  store <4 x i16> %tmp3, ptr %arrayidx8
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  br label %for.cond

for.end:
  ret void
}


; CHECK:      Arrays {
; CHECK-NEXT:     <4 x i16> MemRef_A[*]; // Element size 8
; CHECK-NEXT: }
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [N, P] -> { Stmt_for_body[i0] : 0 <= i0 < N };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [N, P] -> { Stmt_for_body[i0] -> [i0] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N, P] -> { Stmt_for_body[i0] -> MemRef_A[P + 7i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [N, P] -> { Stmt_for_body[i0] -> MemRef_A[7i0] };
; CHECK-NEXT: }
