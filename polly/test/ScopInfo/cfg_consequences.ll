; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; void consequences(int *A, int bool_cond, int lhs, int rhs) {
;
;   BC: *A = 0;
;     if (bool_cond)
;   S_BC:     *A = 0;
;   M_BC: *A = 0;
;
;   NEG_BC: *A = 0;
;     if (!bool_cond)
;   S_NEG_BC: *A = 0;
;   M_NEG_BC: *A = 0;
;
;   SLT: *A = 0;
;     if (lhs < rhs)
;   S_SLT:    *A = 0;
;   M_SLT: *A = 0;
;
;   SLE: *A = 0;
;     if (lhs <= rhs)
;   S_SLE:    *A = 0;
;   M_SLE: *A = 0;
;
;   SGT: *A = 0;
;     if (lhs > rhs)
;   S_SGT:    *A = 0;
;   M_SGT: *A = 0;
;
;   SGE: *A = 0;
;     if (lhs >= rhs)
;   S_SGE:    *A = 0;
;   M_SGE: *A = 0;
;
;   EQ: *A = 0;
;     if (lhs == rhs)
;   S_EQ:    *A = 0;
;   M_EQ: *A = 0;
;
;   NEQ: *A = 0;
;     if (lhs != rhs)
;   S_NEQ:   *A = 0;
;   M_NEQ: *A = 0;
;
; }
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_BC
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_BC[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_BC[] -> [0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_BC[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_S_BC
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_BC[] : bool_cond < 0 or bool_cond > 0 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_BC[] -> [1] : bool_cond < 0 or bool_cond > 0 };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_BC[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_M_BC
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_BC[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_BC[] -> [2] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_BC[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_NEG_BC
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_NEG_BC[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_NEG_BC[] -> [3] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_NEG_BC[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_S_NEG_BC
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_NEG_BC[] : bool_cond = 0 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_NEG_BC[] -> [4] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_NEG_BC[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_M_NEG_BC
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_NEG_BC[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_NEG_BC[] -> [5] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_NEG_BC[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_SLT
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_SLT[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_SLT[] -> [6] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_SLT[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_S_SLT
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_SLT[] : rhs > lhs };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_SLT[] -> [7] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_SLT[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_M_SLT
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_SLT[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_SLT[] -> [8] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_SLT[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_SLE
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_SLE[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_SLE[] -> [9] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_SLE[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_S_SLE
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_SLE[] : rhs >= lhs };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_SLE[] -> [10] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_SLE[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_M_SLE
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_SLE[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_SLE[] -> [11] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_SLE[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_SGT
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_SGT[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_SGT[] -> [12] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_SGT[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_S_SGT
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_SGT[] : rhs < lhs };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_SGT[] -> [13] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_SGT[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_M_SGT
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_SGT[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_SGT[] -> [14] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_SGT[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_SGE
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_SGE[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_SGE[] -> [15] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_SGE[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_S_SGE
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_SGE[] : rhs <= lhs };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_SGE[] -> [16] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_SGE[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_M_SGE
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_SGE[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_SGE[] -> [17] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_SGE[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_EQ
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_EQ[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_EQ[] -> [18] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_EQ[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_S_EQ
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_EQ[] : rhs = lhs };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_EQ[] -> [19] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_EQ[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_M_EQ
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_EQ[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_EQ[] -> [20] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_EQ[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_NEQ
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_NEQ[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_NEQ[] -> [21] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_NEQ[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_S_NEQ
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_NEQ[] : rhs > lhs or rhs < lhs };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_NEQ[] -> [22] : rhs > lhs or rhs < lhs };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_S_NEQ[] -> MemRef_A[0] };
; CHECK-NEXT:     Stmt_M_NEQ
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_NEQ[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_NEQ[] -> [23] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [bool_cond, lhs, rhs] -> { Stmt_M_NEQ[] -> MemRef_A[0] };
; CHECK-NEXT: }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @consequences(ptr %A, i32 %bool_cond, i32 %lhs, i32 %rhs) {
entry:
  br label %BC

BC:                                               ; preds = %entry
  store i32 0, ptr %A, align 4
  %tobool = icmp eq i32 %bool_cond, 0
  br i1 %tobool, label %M_BC, label %S_BC

S_BC:                                             ; preds = %if.then
  store i32 0, ptr %A, align 4
  br label %M_BC

M_BC:                                           ; preds = %BC, %S_BC
  store i32 0, ptr %A, align 4
  br label %NEG_BC

NEG_BC:                                           ; preds = %if.end
  store i32 0, ptr %A, align 4
  %tobool1 = icmp eq i32 %bool_cond, 0
  br i1 %tobool1, label %S_NEG_BC, label %M_NEG_BC

S_NEG_BC:                                         ; preds = %if.then.2
  store i32 0, ptr %A, align 4
  br label %M_NEG_BC

M_NEG_BC:                                         ; preds = %NEG_BC, %S_NEG_BC
  store i32 0, ptr %A, align 4
  br label %SLT

SLT:                                              ; preds = %if.end.3
  store i32 0, ptr %A, align 4
  %cmp = icmp slt i32 %lhs, %rhs
  br i1 %cmp, label %S_SLT, label %M_SLT

S_SLT:                                            ; preds = %if.then.4
  store i32 0, ptr %A, align 4
  br label %M_SLT

M_SLT:                                         ; preds = %S_SLT, %SLT
  store i32 0, ptr %A, align 4
  br label %SLE

SLE:                                              ; preds = %if.end.5
  store i32 0, ptr %A, align 4
  %cmp6 = icmp sgt i32 %lhs, %rhs
  br i1 %cmp6, label %M_SLE, label %S_SLE

S_SLE:                                            ; preds = %if.then.7
  store i32 0, ptr %A, align 4
  br label %M_SLE

M_SLE:                                         ; preds = %SLE, %S_SLE
  store i32 0, ptr %A, align 4
  br label %SGT

SGT:                                              ; preds = %if.end.8
  store i32 0, ptr %A, align 4
  %cmp9 = icmp sgt i32 %lhs, %rhs
  br i1 %cmp9, label %S_SGT, label %M_SGT

S_SGT:                                            ; preds = %if.then.10
  store i32 0, ptr %A, align 4
  br label %M_SGT

M_SGT:                                        ; preds = %S_SGT, %SGT
  store i32 0, ptr %A, align 4
  br label %SGE

SGE:                                              ; preds = %if.end.11
  store i32 0, ptr %A, align 4
  %cmp12 = icmp slt i32 %lhs, %rhs
  br i1 %cmp12, label %M_SGE, label %S_SGE

S_SGE:                                            ; preds = %if.then.13
  store i32 0, ptr %A, align 4
  br label %M_SGE

M_SGE:                                        ; preds = %SGE, %S_SGE
  store i32 0, ptr %A, align 4
  br label %EQ

EQ:                                               ; preds = %if.end.14
  store i32 0, ptr %A, align 4
  %cmp15 = icmp eq i32 %lhs, %rhs
  br i1 %cmp15, label %S_EQ, label %M_EQ

S_EQ:                                             ; preds = %if.then.16
  store i32 0, ptr %A, align 4
  br label %M_EQ

M_EQ:                                        ; preds = %S_EQ, %EQ
  store i32 0, ptr %A, align 4
  br label %NEQ

NEQ:                                              ; preds = %if.end.17
  store i32 0, ptr %A, align 4
  %cmp18 = icmp eq i32 %lhs, %rhs
  br i1 %cmp18, label %M_NEQ, label %S_NEQ

S_NEQ:                                            ; preds = %if.then.19
  store i32 0, ptr %A, align 4
  br label %M_NEQ

M_NEQ:                                        ; preds = %NEQ, %S_NEQ
  store i32 0, ptr %A, align 4
  br label %exit

exit:
  ret void
}
