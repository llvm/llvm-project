; RUN: opt < %s -mattr=+sve -passes=loop-vectorize -force-vector-interleave=1 -prefer-predicate-over-epilogue=predicate-dont-vectorize -S | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; Tests basic vectorization of scalable homogeneous struct literal returns.

define void @struct_return_f32_widen(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: define void @struct_return_f32_widen
; CHECK-SAME:  (ptr noalias [[IN:%.*]], ptr noalias writeonly [[OUT_A:%.*]], ptr noalias writeonly [[OUT_B:%.*]])
; CHECK:       vector.body:
; CHECK:         [[WIDE_CALL:%.*]] = call { <vscale x 4 x float>, <vscale x 4 x float> } @scalable_vec_masked_foo(<vscale x 4 x float> [[WIDE_MASKED_LOAD:%.*]], <vscale x 4 x i1> [[ACTIVE_LANE_MASK:%.*]])
; CHECK:         [[WIDE_A:%.*]] = extractvalue { <vscale x 4 x float>, <vscale x 4 x float> } [[WIDE_CALL]], 0
; CHECK:         [[WIDE_B:%.*]] = extractvalue { <vscale x 4 x float>, <vscale x 4 x float> } [[WIDE_CALL]], 1
; CHECK:         call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> [[WIDE_A]], ptr {{%.*}}, i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK]])
; CHECK:         call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> [[WIDE_B]], ptr {{%.*}}, i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK]])
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

define void @struct_return_f64_widen(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: define void @struct_return_f64_widen
; CHECK-SAME:  (ptr noalias [[IN:%.*]], ptr noalias writeonly [[OUT_A:%.*]], ptr noalias writeonly [[OUT_B:%.*]])
; CHECK:       vector.body:
; CHECK:         [[WIDE_CALL:%.*]] = call { <vscale x 2 x double>, <vscale x 2 x double> } @scalable_vec_masked_bar(<vscale x 2 x double> [[WIDE_MASKED_LOAD:%.*]], <vscale x 2 x i1> [[ACTIVE_LANE_MASK:%.*]])
; CHECK:         [[WIDE_A:%.*]] = extractvalue { <vscale x 2 x double>, <vscale x 2 x double> } [[WIDE_CALL]], 0
; CHECK:         [[WIDE_B:%.*]] = extractvalue { <vscale x 2 x double>, <vscale x 2 x double> } [[WIDE_CALL]], 1
; CHECK:         call void @llvm.masked.store.nxv2f64.p0(<vscale x 2 x double> [[WIDE_A]], ptr {{%.*}}, i32 8, <vscale x 2 x i1> [[ACTIVE_LANE_MASK]])
; CHECK:         call void @llvm.masked.store.nxv2f64.p0(<vscale x 2 x double> [[WIDE_B]], ptr {{%.*}}, i32 8, <vscale x 2 x i1> [[ACTIVE_LANE_MASK]])
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %in, i64 %iv
  %in_val = load double, ptr %arrayidx, align 8
  %call = tail call { double, double } @bar(double %in_val) #1
  %extract_a = extractvalue { double, double } %call, 0
  %extract_b = extractvalue { double, double } %call, 1
  %arrayidx2 = getelementptr inbounds double, ptr %out_a, i64 %iv
  store double %extract_a, ptr %arrayidx2, align 8
  %arrayidx4 = getelementptr inbounds double, ptr %out_b, i64 %iv
  store double %extract_b, ptr %arrayidx4, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

define void @struct_return_f32_widen_rt_checks(ptr %in, ptr writeonly %out_a, ptr writeonly %out_b) {
; CHECK-LABEL: define void @struct_return_f32_widen_rt_checks
; CHECK-SAME:  (ptr [[IN:%.*]], ptr writeonly [[OUT_A:%.*]], ptr writeonly [[OUT_B:%.*]])
; CHECK:       entry:
; CHECK:         br i1 false, label %scalar.ph, label %vector.memcheck
; CHECK:       vector.memcheck:
; CHECK:       vector.body:
; CHECK:         call { <vscale x 4 x float>, <vscale x 4 x float> } @scalable_vec_masked_foo(<vscale x 4 x float> [[WIDE_MASKED_LOAD:%.*]], <vscale x 4 x i1> [[ACTIVE_LANE_MASK:%.*]])
; CHECK:       for.body:
; CHECK:         call { float, float } @foo(float [[LOAD:%.*]])
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
declare { double, double } @bar(double)

declare { <vscale x 4 x float>, <vscale x 4 x float> } @scalable_vec_masked_foo(<vscale x 4 x float>, <vscale x 4 x i1>)
declare { <vscale x 2 x double>, <vscale x 2 x double> } @scalable_vec_masked_bar(<vscale x 2 x double>, <vscale x 2 x i1>)


attributes #0 = { nounwind "vector-function-abi-variant"="_ZGVsMxv_foo(scalable_vec_masked_foo)" }
attributes #1 = { nounwind "vector-function-abi-variant"="_ZGVsMxv_bar(scalable_vec_masked_bar)" }
