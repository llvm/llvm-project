; REQUIRES: asserts
; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -force-vector-width=2 -force-vector-interleave=1 -debug-only=loop-vectorize -disable-output -S 2>&1 | FileCheck %s

define void @struct_return_f32_widen(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: LV: Checking a loop in 'struct_return_f32_widen'
; CHECK:       VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT:  Live-in vp<%0> = VF * UF
; CHECK-NEXT:  Live-in vp<%1> = vector-trip-count
; CHECK-NEXT:  Live-in ir<1024> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:  Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT:  <x1> vector loop: {
; CHECK-NEXT:    vector.body:
; CHECK-NEXT:      EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%7>
; CHECK-NEXT:      vp<%3> = SCALAR-STEPS vp<%2>, ir<1>
; CHECK-NEXT:      CLONE ir<%arrayidx> = getelementptr inbounds ir<%in>, vp<%3>
; CHECK-NEXT:      vp<%4> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:      WIDEN ir<%in_val> = load vp<%4>
; CHECK-NEXT:      WIDEN-CALL ir<%call>, ir<%call>.1 = call @foo(ir<%in_val>) (using library function: fixed_vec_foo)
; CHECK-NEXT:      CLONE ir<%arrayidx2> = getelementptr inbounds ir<%out_a>, vp<%3>
; CHECK-NEXT:      vp<%5> = vector-pointer ir<%arrayidx2>
; CHECK-NEXT:      WIDEN store vp<%5>, ir<%call>
; CHECK-NEXT:      CLONE ir<%arrayidx4> = getelementptr inbounds ir<%out_b>, vp<%3>
; CHECK-NEXT:      vp<%6> = vector-pointer ir<%arrayidx4>
; CHECK-NEXT:      WIDEN store vp<%6>, ir<%call>.1
; CHECK-NEXT:      EMIT vp<%7> = add nuw vp<%2>, vp<%0>
; CHECK-NEXT:      EMIT branch-on-count vp<%7>, vp<%1>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %iv
  %in_val = load float, ptr %arrayidx, align 4
  %call = tail call { float, float } @foo(float %in_val) #0
  %extract_a = extractvalue { float, float } %call, 0
  %extract_b = extractvalue { float, float } %call, 1
  %arrayidx2 = getelementptr inbounds float, ptr %out_a, i64 %iv
  store float %extract_a, ptr %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds float, ptr %out_b, i64 %iv
  store float %extract_b, ptr %arrayidx4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

declare { float, float } @foo(float)

declare { <2 x float>, <2 x float> } @fixed_vec_foo(<2 x float>)

attributes #0 = { nounwind "vector-function-abi-variant"="_ZGVnN2v_foo(fixed_vec_foo)" }
