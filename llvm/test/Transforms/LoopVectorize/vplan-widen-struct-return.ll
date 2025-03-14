; REQUIRES: asserts
; RUN: opt < %s -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 -debug-only=loop-vectorize -disable-output -S 2>&1 | FileCheck %s

define void @struct_return_f32_widen(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: LV: Checking a loop in 'struct_return_f32_widen'
; CHECK:       VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT:  Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT:  Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT:  Live-in ir<1024> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:  Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:  Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT:  <x1> vector loop: {
; CHECK-NEXT:    vector.body:
; CHECK-NEXT:      EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:      vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:      CLONE ir<%arrayidx> = getelementptr inbounds ir<%in>, vp<[[STEPS]]>
; CHECK-NEXT:      vp<[[IN_VEC_PTR:%.+]]> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:      WIDEN ir<%in_val> = load vp<[[IN_VEC_PTR]]>
; CHECK-NEXT:      WIDEN-CALL ir<%call> = call  @foo(ir<%in_val>) (using library function: fixed_vec_foo)
; CHECK-NEXT:      WIDEN ir<%extract_a> = extractvalue ir<%call>, ir<0>
; CHECK-NEXT:      WIDEN ir<%extract_b> = extractvalue ir<%call>, ir<1>
; CHECK-NEXT:      CLONE ir<%arrayidx2> = getelementptr inbounds ir<%out_a>, vp<[[STEPS]]>
; CHECK-NEXT:      vp<[[OUT_A_VEC_PTR:%.+]]> = vector-pointer ir<%arrayidx2>
; CHECK-NEXT:      WIDEN store vp<[[OUT_A_VEC_PTR]]>, ir<%extract_a>
; CHECK-NEXT:      CLONE ir<%arrayidx4> = getelementptr inbounds ir<%out_b>, vp<[[STEPS]]>
; CHECK-NEXT:      vp<[[OUT_B_VEC_PTR:%.+]]> = vector-pointer ir<%arrayidx4>
; CHECK-NEXT:      WIDEN store vp<[[OUT_B_VEC_PTR]]>, ir<%extract_b>
; CHECK-NEXT:      EMIT vp<%index.next> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:      EMIT branch-on-count vp<%index.next>, vp<[[VTC]]>
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

define void @struct_return_f32_replicate(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: LV: Checking a loop in 'struct_return_f32_replicate'
; CHECK:       VPlan 'Initial VPlan for VF={2},UF>=1' {
; CHECK-NEXT:  Live-in vp<[[VFxUF:%.+]]> = VF * UF
; CHECK-NEXT:  Live-in vp<[[VTC:%.+]]> = vector-trip-count
; CHECK-NEXT:  Live-in ir<1024> = original trip-count
; CHECK-EMPTY:
; CHECK-NEXT:  ir-bb<entry>:
; CHECK-NEXT:  Successor(s): vector.ph
; CHECK-EMPTY:
; CHECK-NEXT:  vector.ph:
; CHECK-NEXT:  Successor(s): vector loop
; CHECK-EMPTY:
; CHECK-NEXT:  <x1> vector loop: {
; CHECK-NEXT:    vector.body:
; CHECK-NEXT:      EMIT vp<[[CAN_IV:%.+]]> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK-NEXT:      vp<[[STEPS:%.+]]> = SCALAR-STEPS vp<[[CAN_IV]]>, ir<1>
; CHECK-NEXT:      CLONE ir<%arrayidx> = getelementptr inbounds ir<%in>, vp<[[STEPS]]>
; CHECK-NEXT:      vp<[[IN_VEC_PTR:%.+]]> = vector-pointer ir<%arrayidx>
; CHECK-NEXT:      WIDEN ir<%in_val> = load vp<[[IN_VEC_PTR]]>
; CHECK-NEXT:      REPLICATE ir<%call> = call @foo(ir<%in_val>)
; CHECK-NEXT:      WIDEN ir<%extract_a> = extractvalue ir<%call>, ir<0>
; CHECK-NEXT:      WIDEN ir<%extract_b> = extractvalue ir<%call>, ir<1>
; CHECK-NEXT:      CLONE ir<%arrayidx2> = getelementptr inbounds ir<%out_a>, vp<[[STEPS]]>
; CHECK-NEXT:      vp<[[OUT_A_VEC_PTR:%.+]]> = vector-pointer ir<%arrayidx2>
; CHECK-NEXT:      WIDEN store vp<[[OUT_A_VEC_PTR]]>, ir<%extract_a>
; CHECK-NEXT:      CLONE ir<%arrayidx4> = getelementptr inbounds ir<%out_b>, vp<[[STEPS]]>
; CHECK-NEXT:      vp<[[OUT_B_VEC_PTR:%.+]]> = vector-pointer ir<%arrayidx4>
; CHECK-NEXT:      WIDEN store vp<[[OUT_B_VEC_PTR]]>, ir<%extract_b>
; CHECK-NEXT:      EMIT vp<%index.next> = add nuw vp<[[CAN_IV]]>, vp<[[VFxUF]]>
; CHECK-NEXT:      EMIT branch-on-count vp<%index.next>, vp<[[VTC]]>
; CHECK-NEXT:    No successors
; CHECK-NEXT:  }
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %iv
  %in_val = load float, ptr %arrayidx, align 4
  ; #3 does not have a fixed-size vector mapping (so replication is used)
  %call = tail call { float, float } @foo(float %in_val) #1
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
declare { <vscale x 4 x float>, <vscale x 4 x float> } @scalable_vec_masked_foo(<vscale x 4 x float>, <vscale x 4 x i1>)

attributes #0 = { nounwind "vector-function-abi-variant"="_ZGVnN2v_foo(fixed_vec_foo)" }
attributes #1 = { nounwind "vector-function-abi-variant"="_ZGVsMxv_foo(scalable_vec_masked_foo)" }
