; RUN: opt %loadPolly -polly-ignore-aliasing -polly-invariant-load-hoisting=true -polly-scops -polly-print-import-jscop -polly-codegen -disable-output < %s | FileCheck %s
;
; This violated an assertion in setNewAccessRelation that assumed base pointers
; to be load-hoisted. Without this assertion, it codegen would generate invalid
; code.
;
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

define void @base_pointer_load_is_inst_inside_invariant_1(i64 %n, float** %A) {
entry:
  br label %for.i

for.i:
  %indvar.i = phi i64 [ %indvar.i.next, %for.i.inc ], [ 0, %entry ]
  br label %S1

S1:
  %ptr = load float*, float** %A
  %conv = sitofp i64 %indvar.i to float
  %arrayidx5 = getelementptr float, float* %ptr, i64 %indvar.i
  store float %conv, float* %arrayidx5, align 4
  br label %for.i.inc

for.i.inc:
  %indvar.i.next = add i64 %indvar.i, 1
  %exitcond.i = icmp ne i64 %indvar.i.next, %n
  br i1 %exitcond.i, label %for.i, label %exit

exit:
  ret void
}


; Detected by -polly-detect with required load hoist.
; CHECK-NOT: Valid Region for Scop: for.i => exit
;
; Load hoist if %ptr by -polly-scops.
; CHECK:      Invariant Accesses: {
; CHECK-NEXT:     ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:         [n] -> { Stmt_S1[i0] -> MemRef_A[0] };
; CHECK-NEXT:     Execution Context: [n] -> {  : n > 0 }
; CHECK-NEXT: }
