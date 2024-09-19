; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -polly-invariant-load-hoisting=true -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt %loadNPMPolly -passes=polly-codegen -polly-invariant-load-hoisting=true -S < %s 2>&1 | FileCheck %s --check-prefix=CODEGEN
;
;    int U;
;    int f(int *A) {
;      int i = 0, x, y;
;      do {
;        x = (*(int *)&U);
;        y = (int)(*(float *)&U);
;        A[i] = x + y;
;      } while (i++ < 100);
;      return x + y;
;    }
;
; CHECK:      Invariant Accesses: {
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_do_body[i0] -> MemRef_U[0] };
; CHECK-NEXT:         Execution Context: {  :  }
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_do_body[i0] -> MemRef_U[0] };
; CHECK-NEXT:         Execution Context: {  :  }
; CHECK-NEXT: }
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_do_body
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_do_body[i0] : 0 <= i0 <= 100 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_do_body[i0] -> [i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             { Stmt_do_body[i0] -> MemRef_A[i0] };
; CHECK-NEXT: }
;
; CODEGEN: entry:
; CODEGEN-DAG:   %U.f.preload.s2a = alloca float
; CODEGEN-DAG:   %U.i.preload.s2a = alloca i32
; CODEGEN:   br label %polly.split_new_and_old
;
; CODEGEN: polly.preload.begin:
; CODEGEN-DAG:   %U.load[[f:[.0-9]*]] = load float, ptr @U
; CODEGEN-DAG:   store float %U.load[[f]], ptr %U.f.preload.s2a
; CODEGEN-DAG:   %U.load[[i:[.0-9]*]] = load i32, ptr @U
; CODEGEN-DAG:   store i32 %U.load[[i]], ptr %U.i.preload.s2a
;
; CODEGEN:     polly.merge_new_and_old:
; CODEGEN-DAG:   %U.f.merge = phi float [ %U.f.final_reload, %polly.exiting ], [ %U.f, %do.cond ]
; CODEGEN-DAG:   %U.i.merge = phi i32 [ %U.i.final_reload, %polly.exiting ], [ %U.i, %do.cond ]
;
; CODEGEN: polly.loop_exit:
; CODEGEN-DAG:   %U.f.final_reload = load float, ptr %U.f.preload.s2a
; CODEGEN-DAG:   %U.i.final_reload = load i32, ptr %U.i.preload.s2a
;
; CODEGEN: polly.stmt.do.body:
; CODEGEN:   %p_add = add nsw i32 %U.load[[i]], %p_conv
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@U = common global i32 0, align 4

define i32 @f(ptr %A) {
entry:
  br label %do.body

do.body:                                          ; preds = %do.cond, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %do.cond ], [ 0, %entry ]
  %U.i = load i32, ptr @U, align 4
  %U.f = load float, ptr @U, align 4
  %conv = fptosi float %U.f to i32
  %add = add nsw i32 %U.i, %conv
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %add, ptr %arrayidx, align 4
  br label %do.cond

do.cond:                                          ; preds = %do.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 101
  br i1 %exitcond, label %do.body, label %do.end

do.end:                                           ; preds = %do.cond
  %conv2 = fptosi float %U.f to i32
  %add2 = add nsw i32 %U.i, %conv2
  ret i32 %add2
}
