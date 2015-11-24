; RUN: opt %loadPolly -polly-detect -analyze < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

declare float* @getNextBasePtr(float*) readnone nounwind

define void @base_pointer_is_inst_inside_invariant_1(i64 %n, float* %A, float* %B) {
entry:
  br label %for.i

for.i:
  %indvar.i = phi i64 [ %indvar.i.next, %for.i.inc ], [ 0, %entry ]
  br label %S1

S1:
; To get an instruction inside a region, we use a function without side
; effects on which SCEV blocks, but for which it is still clear that the
; return value remains invariant throughout the whole loop.
  %ptr = call float* @getNextBasePtr(float* %A)
  %conv = sitofp i64 %indvar.i to float
  %arrayidx5 = getelementptr float, float* %ptr, i64 %indvar.i
  store float %conv, float* %arrayidx5, align 4
  store float 1.0, float* %B
  br label %for.i.inc

for.i.inc:
  %indvar.i.next = add i64 %indvar.i, 1
  %exitcond.i = icmp ne i64 %indvar.i.next, %n
  br i1 %exitcond.i, label %for.i, label %exit

exit:
  ret void
}

; CHECK-NOT: Valid Region for Scop
