; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -polly-delinearize=false -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -polly-delinearize=false -polly-allow-nonaffine -disable-output < %s 2>&1 | FileCheck %s --check-prefix=NONAFFINE
; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=DELIN
; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -polly-allow-nonaffine -disable-output < %s 2>&1 | FileCheck %s --check-prefix=DELIN
; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -polly-delinearize=false -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -polly-delinearize=false -polly-allow-nonaffine -disable-output < %s 2>&1 | FileCheck %s --check-prefix=NONAFFINE
; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s --check-prefix=DELIN
; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -polly-allow-nonaffine -disable-output < %s 2>&1 | FileCheck %s --check-prefix=DELIN

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; void single-and-multi-dimensional-array(long n,float X[n][n]) {
;  for (long i1 = 0; i1 < n; i1++)
;    X[i1][0] = 1;
;
;  for (long i2 = 0; i2 < n; i2++)
;    X[n-1][i2] = 1;
; }
;
; In previous versions of Polly, the second access was detected as single
; dimensional access whereas the first one was detected as multi-dimensional.
; This test case checks that we now consistently delinearize the array accesses.

; CHECK-NOT: Stmt_for_i_1

; NONAFFINE:      p0: %n
; NONAFFINE-NEXT: p1: ((-1 + %n) * %n)
;
; NONAFFINE:      Statements {
; NONAFFINE-NEXT:     Stmt_for_i_1
; NONAFFINE-NEXT:         Domain :=
; NONAFFINE-NEXT:             [n, p_1] -> { Stmt_for_i_1[i0] : 0 <= i0 < n };
; NONAFFINE-NEXT:         Schedule :=
; NONAFFINE-NEXT:             [n, p_1] -> { Stmt_for_i_1[i0] -> [0, i0] };
; NONAFFINE-NEXT:         MayWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; NONAFFINE-NEXT:             [n, p_1] -> { Stmt_for_i_1[i0] -> MemRef_X[o0] };
; NONAFFINE-NEXT:     Stmt_for_i_2
; NONAFFINE-NEXT:         Domain :=
; NONAFFINE-NEXT:             [n, p_1] -> { Stmt_for_i_2[i0] : 0 <= i0 < n };
; NONAFFINE-NEXT:         Schedule :=
; NONAFFINE-NEXT:             [n, p_1] -> { Stmt_for_i_2[i0] -> [1, i0] };
; NONAFFINE-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; NONAFFINE-NEXT:             [n, p_1] -> { Stmt_for_i_2[i0] -> MemRef_X[p_1 + i0] };
; NONAFFINE-NEXT: }

; DELIN:      Statements {
; DELIN-NEXT:     Stmt_for_i_1
; DELIN-NEXT:         Domain :=
; DELIN-NEXT:             [n] -> { Stmt_for_i_1[i0] : 0 <= i0 < n };
; DELIN-NEXT:         Schedule :=
; DELIN-NEXT:             [n] -> { Stmt_for_i_1[i0] -> [0, i0] };
; DELIN-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; DELIN-NEXT:             [n] -> { Stmt_for_i_1[i0] -> MemRef_X[i0, 0] };
; DELIN-NEXT:     Stmt_for_i_2
; DELIN-NEXT:         Domain :=
; DELIN-NEXT:             [n] -> { Stmt_for_i_2[i0] : 0 <= i0 < n };
; DELIN-NEXT:         Schedule :=
; DELIN-NEXT:             [n] -> { Stmt_for_i_2[i0] -> [1, i0] };
; DELIN-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; DELIN-NEXT:             [n] -> { Stmt_for_i_2[i0] -> MemRef_X[-1 + n, i0] };
; DELIN-NEXT: }

define void @single-and-multi-dimensional-array(i64 %n, ptr %X) {
entry:
  br label %for.i.1

for.i.1:
  %indvar.1 = phi i64 [ 0, %entry ], [ %indvar.next.1, %for.i.1 ]
  %offset.1 = mul i64 %n, %indvar.1
  %arrayidx.1 = getelementptr float, ptr %X, i64 %offset.1
  store float 1.000000e+00, ptr %arrayidx.1
  %indvar.next.1 = add nsw i64 %indvar.1, 1
  %exitcond.1 = icmp ne i64 %indvar.next.1, %n
  br i1 %exitcond.1, label %for.i.1, label %next

next:
  br label %for.i.2

for.i.2:
  %indvar.2 = phi i64 [ 0, %next ], [ %indvar.next.2, %for.i.2 ]
  %offset.2.a = add i64 %n, -1
  %offset.2.b = mul i64 %n, %offset.2.a
  %offset.2.c = add i64 %offset.2.b, %indvar.2
  %arrayidx.2 = getelementptr float, ptr %X, i64 %offset.2.c
  store float 1.000000e+00, ptr %arrayidx.2
  %indvar.next.2 = add nsw i64 %indvar.2, 1
  %exitcond.2 = icmp ne i64 %indvar.next.2, %n
  br i1 %exitcond.2, label %for.i.2, label %exit

exit:
  ret void
}
