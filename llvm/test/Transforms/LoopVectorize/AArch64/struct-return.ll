; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -force-vector-interleave=1 -S | FileCheck %s --check-prefixes=NEON
; RUN: opt < %s -mattr=+sve -passes=loop-vectorize,dce,instcombine -force-vector-interleave=1 -S -prefer-predicate-over-epilogue=predicate-dont-vectorize | FileCheck %s --check-prefixes=SVE_TF

target triple = "aarch64-unknown-linux-gnu"

; Tests basic vectorization of homogeneous struct literal returns.

define void @struct_return_f32_widen(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; NEON-LABEL: define void @struct_return_f32_widen
; NEON-SAME:  (ptr noalias [[IN:%.*]], ptr noalias writeonly [[OUT_A:%.*]], ptr noalias writeonly [[OUT_B:%.*]])
; NEON:       vector.body:
; NEON:         [[WIDE_CALL:%.*]] = call { <4 x float>, <4 x float> } @fixed_vec_foo(<4 x float> [[WIDE_LOAD:%.*]])
; NEON:         [[WIDE_A:%.*]] = extractvalue { <4 x float>, <4 x float> } [[WIDE_CALL]], 0
; NEON:         [[WIDE_B:%.*]] = extractvalue { <4 x float>, <4 x float> } [[WIDE_CALL]], 1
; NEON:         store <4 x float> [[WIDE_A]], ptr {{%.*}}, align 4
; NEON:         store <4 x float> [[WIDE_B]], ptr {{%.*}}, align 4
;
; SVE_TF-LABEL: define void @struct_return_f32_widen
; SVE_TF-SAME:  (ptr noalias [[IN:%.*]], ptr noalias writeonly [[OUT_A:%.*]], ptr noalias writeonly [[OUT_B:%.*]])
; SVE_TF:       vector.body:
; SVE_TF:         [[WIDE_CALL:%.*]] = call { <vscale x 4 x float>, <vscale x 4 x float> } @scalable_vec_masked_foo(<vscale x 4 x float> [[WIDE_MASKED_LOAD:%.*]], <vscale x 4 x i1> [[ACTIVE_LANE_MASK:%.*]])
; SVE_TF:         [[WIDE_A:%.*]] = extractvalue { <vscale x 4 x float>, <vscale x 4 x float> } [[WIDE_CALL]], 0
; SVE_TF:         [[WIDE_B:%.*]] = extractvalue { <vscale x 4 x float>, <vscale x 4 x float> } [[WIDE_CALL]], 1
; SVE_TF:         call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> [[WIDE_A]], ptr {{%.*}}, i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK]])
; SVE_TF:         call void @llvm.masked.store.nxv4f32.p0(<vscale x 4 x float> [[WIDE_B]], ptr {{%.*}}, i32 4, <vscale x 4 x i1> [[ACTIVE_LANE_MASK]])
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %indvars.iv
  %in_val = load float, ptr %arrayidx, align 4
  %call = tail call { float, float } @foo(float %in_val) #0
  %extract_a = extractvalue { float, float } %call, 0
  %extract_b = extractvalue { float, float } %call, 1
  %arrayidx2 = getelementptr inbounds float, ptr %out_a, i64 %indvars.iv
  store float %extract_a, ptr %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds float, ptr %out_b, i64 %indvars.iv
  store float %extract_b, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

define void @struct_return_f64_widen(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; NEON-LABEL: define void @struct_return_f64_widen
; NEON-SAME:  (ptr noalias [[IN:%.*]], ptr noalias writeonly [[OUT_A:%.*]], ptr noalias writeonly [[OUT_B:%.*]])
; NEON:        vector.body:
; NEON:          [[WIDE_CALL:%.*]] = call { <2 x double>, <2 x double> } @fixed_vec_bar(<2 x double> [[WIDE_LOAD:%.*]])
; NEON:          [[WIDE_A:%.*]] = extractvalue { <2 x double>, <2 x double> } [[WIDE_CALL]], 0
; NEON:          [[WIDE_B:%.*]] = extractvalue { <2 x double>, <2 x double> } [[WIDE_CALL]], 1
; NEON:          store <2 x double> [[WIDE_A]], ptr {{%.*}}, align 8
; NEON:          store <2 x double> [[WIDE_B]], ptr {{%.*}}, align 8
;
; SVE_TF-LABEL: define void @struct_return_f64_widen
; SVE_TF-SAME:  (ptr noalias [[IN:%.*]], ptr noalias writeonly [[OUT_A:%.*]], ptr noalias writeonly [[OUT_B:%.*]])
; SVE_TF:       vector.body:
; SVE_TF:         [[WIDE_CALL:%.*]] = call { <vscale x 2 x double>, <vscale x 2 x double> } @scalable_vec_masked_bar(<vscale x 2 x double> [[WIDE_MASKED_LOAD:%.*]], <vscale x 2 x i1> [[ACTIVE_LANE_MASK:%.*]])
; SVE_TF:         [[WIDE_A:%.*]] = extractvalue { <vscale x 2 x double>, <vscale x 2 x double> } [[WIDE_CALL]], 0
; SVE_TF:         [[WIDE_B:%.*]] = extractvalue { <vscale x 2 x double>, <vscale x 2 x double> } [[WIDE_CALL]], 1
; SVE_TF:         call void @llvm.masked.store.nxv2f64.p0(<vscale x 2 x double> [[WIDE_A]], ptr {{%.*}}, i32 8, <vscale x 2 x i1> [[ACTIVE_LANE_MASK]])
; SVE_TF:         call void @llvm.masked.store.nxv2f64.p0(<vscale x 2 x double> [[WIDE_B]], ptr {{%.*}}, i32 8, <vscale x 2 x i1> [[ACTIVE_LANE_MASK]])
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %in, i64 %indvars.iv
  %in_val = load double, ptr %arrayidx, align 8
  %call = tail call { double, double } @bar(double %in_val) #1
  %extract_a = extractvalue { double, double } %call, 0
  %extract_b = extractvalue { double, double } %call, 1
  %arrayidx2 = getelementptr inbounds double, ptr %out_a, i64 %indvars.iv
  store double %extract_a, ptr %arrayidx2, align 8
  %arrayidx4 = getelementptr inbounds double, ptr %out_b, i64 %indvars.iv
  store double %extract_b, ptr %arrayidx4, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

define void @struct_return_f32_replicate(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; NEON-LABEL: define void @struct_return_f32_replicate
; NEON-SAME:  (ptr noalias [[IN:%.*]], ptr noalias writeonly [[OUT_A:%.*]], ptr noalias writeonly [[OUT_B:%.*]])
; NEON:       vector.body:
; NEON:         [[CALL_LANE_0:%.*]] = tail call { float, float } @foo(float {{%.*}})
; NEON:         [[CALL_LANE_1:%.*]] = tail call { float, float } @foo(float {{%.*}})
; NEON:         [[LANE_0_A:%.*]] = extractvalue { float, float } [[CALL_LANE_0]], 0
; NEON:         [[TMP_A:%.*]] = insertelement <2 x float> poison, float [[LANE_0_A]], i64 0
; NEON:         [[LANE_0_B:%.*]] = extractvalue { float, float } [[CALL_LANE_0]], 1
; NEON:         [[TMP_B:%.*]] = insertelement <2 x float> poison, float [[LANE_0_B]], i64 0
; NEON:         [[LANE_1_A:%.*]] = extractvalue { float, float } [[CALL_LANE_1]], 0
; NEON:         [[WIDE_A:%.*]] = insertelement <2 x float> [[TMP_A]], float [[LANE_1_A]], i64 1
; NEON:         [[LANE_1_B:%.*]] = extractvalue { float, float } [[CALL_LANE_1]], 1
; NEON:         [[WIDE_B:%.*]] = insertelement <2 x float> [[TMP_B]], float [[LANE_1_B]], i64 1
; NEON:         store <2 x float> [[WIDE_A]], ptr {{%.*}}, align 4
; NEON:         store <2 x float> [[WIDE_B]], ptr {{%.*}}, align 4
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %indvars.iv
  %in_val = load float, ptr %arrayidx, align 4
  ; #3 does not have a fixed-size vector mapping (so replication is used)
  %call = tail call { float, float } @foo(float %in_val) #3
  %extract_a = extractvalue { float, float } %call, 0
  %extract_b = extractvalue { float, float } %call, 1
  %arrayidx2 = getelementptr inbounds float, ptr %out_a, i64 %indvars.iv
  store float %extract_a, ptr %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds float, ptr %out_b, i64 %indvars.iv
  store float %extract_b, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

define void @struct_return_f32_widen_rt_checks(ptr %in, ptr writeonly %out_a, ptr writeonly %out_b) {
; NEON-LABEL: define void @struct_return_f32_widen_rt_checks
; NEON-SAME:  (ptr [[IN:%.*]], ptr writeonly [[OUT_A:%.*]], ptr writeonly [[OUT_B:%.*]])
; NEON:       entry:
; NEON:         br i1 false, label %scalar.ph, label %vector.memcheck
; NEON:       vector.memcheck:
; NEON:       vector.body:
; NEON:         call { <4 x float>, <4 x float> } @fixed_vec_foo(<4 x float> [[WIDE_LOAD:%.*]])
; NEON:       for.body:
; NEON          call { float, float } @foo(float [[LOAD:%.*]])
;
; SVE_TF-LABEL: define void @struct_return_f32_widen_rt_checks
; SVE_TF-SAME:  (ptr [[IN:%.*]], ptr writeonly [[OUT_A:%.*]], ptr writeonly [[OUT_B:%.*]])
; SVE_TF:       entry:
; SVE_TF:         br i1 false, label %scalar.ph, label %vector.memcheck
; SVE_TF:       vector.memcheck:
; SVE_TF:       vector.body:
; SVE_TF:         call { <vscale x 4 x float>, <vscale x 4 x float> } @scalable_vec_masked_foo(<vscale x 4 x float> [[WIDE_MASKED_LOAD:%.*]], <vscale x 4 x i1> [[ACTIVE_LANE_MASK:%.*]])
; SVE_TF:       for.body:
; SVE_TF:         call { float, float } @foo(float [[LOAD:%.*]])
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %indvars.iv
  %in_val = load float, ptr %arrayidx, align 4
  %call = tail call { float, float } @foo(float %in_val) #0
  %extract_a = extractvalue { float, float } %call, 0
  %extract_b = extractvalue { float, float } %call, 1
  %arrayidx2 = getelementptr inbounds float, ptr %out_a, i64 %indvars.iv
  store float %extract_a, ptr %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds float, ptr %out_b, i64 %indvars.iv
  store float %extract_b, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

; Negative test. Widening structs with mixed element types is not supported.
define void @negative_mixed_element_type_struct_return(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; NEON-LABEL: define void @negative_mixed_element_type_struct_return
; NEON-NOT:   vector.body:
; NEON-NOT:   call {{.*}} @fixed_vec_baz
;
; SVE_TF-LABEL: define void @negative_mixed_element_type_struct_return
; SVE_TF-NOT:   vector.body:
; SVE_TF-NOT:   call {{.*}} @scalable_vec_masked_baz
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %indvars.iv
  %in_val = load float, ptr %arrayidx, align 4
  %call = tail call { float, i32 } @baz(float %in_val) #2
  %extract_a = extractvalue { float, i32 } %call, 0
  %extract_b = extractvalue { float, i32 } %call, 1
  %arrayidx2 = getelementptr inbounds float, ptr %out_a, i64 %indvars.iv
  store float %extract_a, ptr %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr %out_b, i64 %indvars.iv
  store i32 %extract_b, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

%named_struct = type { double, double }

; Negative test. Widening non-literal structs is not supported.
define void @test_named_struct_return(ptr noalias readonly %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; NEON-LABEL: define void @test_named_struct_return
; NEON-NOT:   vector.body:
; NEON-NOT:   call {{.*}} @fixed_vec_bar
;
; SVE_TF-LABEL: define void @test_named_struct_return
; SVE_TF-NOT:   vector.body:
; SVE_TF-NOT:   call {{.*}} @scalable_vec_masked_bar
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %in, i64 %indvars.iv
  %in_val = load double, ptr %arrayidx, align 8
  %call = tail call %named_struct @bar_named(double %in_val) #4
  %extract_a = extractvalue %named_struct %call, 0
  %extract_b = extractvalue %named_struct %call, 1
  %arrayidx2 = getelementptr inbounds double, ptr %out_a, i64 %indvars.iv
  store double %extract_a, ptr %arrayidx2, align 8
  %arrayidx4 = getelementptr inbounds double, ptr %out_b, i64 %indvars.iv
  store double %extract_b, ptr %arrayidx4, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

; TODO: Allow mixed-struct type vectorization and mark overflow intrinsics as trivially vectorizable.
define void @test_overflow_intrinsic(ptr noalias readonly %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; NEON-LABEL: define void @test_overflow_intrinsic
; NEON-NOT:   vector.body:
; SVE_TF-NOT:   @llvm.sadd.with.overflow.v{{.+}}i32
;
; SVE_TF-LABEL: define void @test_overflow_intrinsic
; SVE_TF-NOT:   vector.body:
; SVE_TF-NOT:   @llvm.sadd.with.overflow.v{{.+}}i32
; SVE_TF-NOT:   @llvm.sadd.with.overflow.nxv{{.+}}i32
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %indvars.iv
  %in_val = load i32, ptr %arrayidx, align 4
  %call = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %in_val, i32 %in_val)
  %extract_ret = extractvalue { i32, i1 } %call, 0
  %extract_overflow = extractvalue { i32, i1 } %call, 1
  %zext_overflow = zext i1 %extract_overflow to i8
  %arrayidx2 = getelementptr inbounds i32, ptr %out_a, i64 %indvars.iv
  store i32 %extract_ret, ptr %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds i8, ptr %out_b, i64 %indvars.iv
  store i8 %zext_overflow, ptr %arrayidx4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void
}

declare { float, float } @foo(float)
declare { double, double } @bar(double)
declare { float, i32 } @baz(float)
declare %named_struct @bar_named(double)

declare { <4 x float>, <4 x float> } @fixed_vec_foo(<4 x float>)
declare { <vscale x 4 x float>, <vscale x 4 x float> } @scalable_vec_masked_foo(<vscale x 4 x float>, <vscale x 4 x i1>)

declare { <2 x double>, <2 x double> } @fixed_vec_bar(<2 x double>)
declare { <vscale x 2 x double>, <vscale x 2 x double> } @scalable_vec_masked_bar(<vscale x 2 x double>, <vscale x 2 x i1>)

declare { <4 x float>, <4 x i32> } @fixed_vec_baz(<4 x float>)
declare { <vscale x 4 x float>, <vscale x 4 x i32> } @scalable_vec_masked_baz(<vscale x 4 x float>, <vscale x 4 x i1>)

attributes #0 = { nounwind "vector-function-abi-variant"="_ZGVnN4v_foo(fixed_vec_foo),_ZGVsMxv_foo(scalable_vec_masked_foo)" }
attributes #1 = { nounwind "vector-function-abi-variant"="_ZGVnN2v_bar(fixed_vec_bar),_ZGVsMxv_bar(scalable_vec_masked_bar)" }
attributes #2 = { nounwind "vector-function-abi-variant"="_ZGVnN4v_baz(fixed_vec_baz),_ZGVsMxv_foo(scalable_vec_masked_baz)" }
attributes #3 = { nounwind "vector-function-abi-variant"="_ZGVsMxv_foo(scalable_vec_masked_foo)" }
attributes #4 = { nounwind "vector-function-abi-variant"="_ZGVnN4v_bar_named(fixed_vec_bar),_ZGVsMxv_bar_named(scalable_vec_masked_bar)" }
