; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s

; ModuleID = 'scalar_to_array.ll'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

@A = common global [1024 x float] zeroinitializer, align 8

; Terminating loops without side-effects will be optimized away, hence
; detecting a scop would be pointless.
; CHECK-NOT: Function: empty
; Function Attrs: nounwind
define i32 @empty() #0 {
entry:
  fence seq_cst
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvar = phi i64 [ %indvar.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvar, 1024
  br i1 %exitcond, label %for.body, label %return

for.body:                                         ; preds = %for.cond
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvar.next = add i64 %indvar, 1
  br label %for.cond

return:                                           ; preds = %for.cond
  fence seq_cst
  ret i32 0
}

; CHECK-LABEL: Function: array_access
; Function Attrs: nounwind
define i32 @array_access() #0 {
entry:
  fence seq_cst
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvar = phi i64 [ %indvar.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvar, 1024
  br i1 %exitcond, label %for.body, label %return

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr [1024 x float], ptr @A, i64 0, i64 %indvar
  %float = uitofp i64 %indvar to float
  store float %float, ptr %arrayidx
  br label %for.inc
; CHECK:     Stmt_for_body
; CHECK-NOT:     ReadAccess
; CHECK:         MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:        { Stmt_for_body[i0] -> MemRef_A[i0] };

for.inc:                                          ; preds = %for.body
  %indvar.next = add i64 %indvar, 1
  br label %for.cond

return:                                           ; preds = %for.cond
  fence seq_cst
  ret i32 0
}

; Function Attrs: nounwind
; CHECK-LABEL: Function: intra_scop_dep
define i32 @intra_scop_dep() #0 {
entry:
  fence seq_cst
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvar = phi i64 [ %indvar.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvar, 1024
  br i1 %exitcond, label %for.body.a, label %return

for.body.a:                                       ; preds = %for.cond
  %arrayidx = getelementptr [1024 x float], ptr @A, i64 0, i64 %indvar
  %scalar = load float, ptr %arrayidx
  br label %for.body.b
; CHECK:      Stmt_for_body_a
; CHECK:          ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:         { Stmt_for_body_a[i0] -> MemRef_A[i0] };
; CHECK:          MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:         { Stmt_for_body_a[i0] -> MemRef_scalar[] };

for.body.b:                                       ; preds = %for.body.a
  %arrayidx2 = getelementptr [1024 x float], ptr @A, i64 0, i64 %indvar
  %float = uitofp i64 %indvar to float
  %sum = fadd float %scalar, %float
  store float %sum, ptr %arrayidx2
  br label %for.inc
; CHECK:      Stmt_for_body_b
; CHECK:          ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:          { Stmt_for_body_b[i0] -> MemRef_scalar[] };
; CHECK:          MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:         { Stmt_for_body_b[i0] -> MemRef_A[i0] };

for.inc:                                          ; preds = %for.body.b
  %indvar.next = add i64 %indvar, 1
  br label %for.cond

return:                                           ; preds = %for.cond
  fence seq_cst
  ret i32 0
}

; It is not possible to have a scop which accesses a scalar element that is
; a global variable. All global variables are pointers containing possibly
; a single element. Hence they do not need to be handled anyways.
; Please note that this is still required when scalar to array rewriting is
; disabled.

; CHECK-LABEL: Function: use_after_scop
; Function Attrs: nounwind
define i32 @use_after_scop() #0 {
entry:
  %scalar.s2a = alloca float
  fence seq_cst
  br label %for.head

for.head:                                         ; preds = %for.inc, %entry
  %indvar = phi i64 [ %indvar.next, %for.inc ], [ 0, %entry ]
  br label %for.body

for.body:                                         ; preds = %for.head
  %arrayidx = getelementptr [1024 x float], ptr @A, i64 0, i64 %indvar
  %scalar = load float, ptr %arrayidx
  store float %scalar, ptr %scalar.s2a
; Escaped uses are still required to be rewritten to stack variable.
; CHECK:      Stmt_for_body
; CHECK:          ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:         { Stmt_for_body[i0] -> MemRef_A[i0] };
; CHECK:          MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:         { Stmt_for_body[i0] -> MemRef_scalar_s2a[0] };
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp ne i64 %indvar.next, 1024
  br i1 %exitcond, label %for.head, label %for.after

for.after:                                        ; preds = %for.inc
  %scalar.loadoutside = load float, ptr %scalar.s2a
  fence seq_cst
  %return_value = fptosi float %scalar.loadoutside to i32
  br label %return

return:                                           ; preds = %for.after
  ret i32 %return_value
}

; We currently do not transform scalar references, that have only read accesses
; in the scop. There are two reasons for this:
;
;  o We don't introduce additional memory references which may yield to compile
;    time overhead.
;  o For integer values, such a translation may block the use of scalar
;    evolution on those values.
;
; CHECK-LABEL: Function: before_scop
; Function Attrs: nounwind
define i32 @before_scop() #0 {
entry:
  br label %preheader

preheader:                                        ; preds = %entry
  %scalar = fadd float 4.000000e+00, 5.000000e+00
  fence seq_cst
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %preheader
  %indvar = phi i64 [ %indvar.next, %for.inc ], [ 0, %preheader ]
  %exitcond = icmp ne i64 %indvar, 1024
  br i1 %exitcond, label %for.body, label %return

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr [1024 x float], ptr @A, i64 0, i64 %indvar
  store float %scalar, ptr %arrayidx
  br label %for.inc
; CHECK:      Stmt_for_body
; CHECK:          MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:         { Stmt_for_body[i0] -> MemRef_A[i0] };

for.inc:                                          ; preds = %for.body
  %indvar.next = add i64 %indvar, 1
  br label %for.cond

return:                                           ; preds = %for.cond
  fence seq_cst
  ret i32 0
}

attributes #0 = { nounwind }
