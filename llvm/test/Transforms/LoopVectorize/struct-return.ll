; RUN: opt < %s -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 -S -pass-remarks=loop-vectorize -pass-remarks-analysis=loop-vectorize 2>%t | FileCheck %s
; RUN: cat %t | FileCheck --check-prefix=CHECK-REMARKS %s

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

; Tests basic vectorization of homogeneous struct literal returns.

; CHECK-REMARKS: remark: {{.*}} vectorized loop
define void @struct_return_f32_widen(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: define void @struct_return_f32_widen
; CHECK-SAME:  (ptr noalias [[IN:%.*]], ptr noalias writeonly [[OUT_A:%.*]], ptr noalias writeonly [[OUT_B:%.*]])
; CHECK:       vector.body:
; CHECK:         [[WIDE_CALL:%.*]] = call { <2 x float>, <2 x float> } @fixed_vec_foo(<2 x float> [[WIDE_LOAD:%.*]])
; CHECK:         [[WIDE_A:%.*]] = extractvalue { <2 x float>, <2 x float> } [[WIDE_CALL]], 0
; CHECK:         [[WIDE_B:%.*]] = extractvalue { <2 x float>, <2 x float> } [[WIDE_CALL]], 1
; CHECK:         store <2 x float> [[WIDE_A]], ptr {{%.*}}, align 4
; CHECK:         store <2 x float> [[WIDE_B]], ptr {{%.*}}, align 4
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

; CHECK-REMARKS: remark: {{.*}} vectorized loop
define void @struct_return_f64_widen(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: define void @struct_return_f64_widen
; CHECK-SAME:  (ptr noalias [[IN:%.*]], ptr noalias writeonly [[OUT_A:%.*]], ptr noalias writeonly [[OUT_B:%.*]])
; CHECK:        vector.body:
; CHECK:          [[WIDE_CALL:%.*]] = call { <2 x double>, <2 x double> } @fixed_vec_bar(<2 x double> [[WIDE_LOAD:%.*]])
; CHECK:          [[WIDE_A:%.*]] = extractvalue { <2 x double>, <2 x double> } [[WIDE_CALL]], 0
; CHECK:          [[WIDE_B:%.*]] = extractvalue { <2 x double>, <2 x double> } [[WIDE_CALL]], 1
; CHECK:          store <2 x double> [[WIDE_A]], ptr {{%.*}}, align 8
; CHECK:          store <2 x double> [[WIDE_B]], ptr {{%.*}}, align 8
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

; CHECK-REMARKS: remark: {{.*}} vectorized loop
; Note: Later instcombines reduce this down quite a lot.
define void @struct_return_f32_replicate(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: define void @struct_return_f32_replicate
; CHECK-SAME:  (ptr noalias [[IN:%.*]], ptr noalias writeonly [[OUT_A:%.*]], ptr noalias writeonly [[OUT_B:%.*]])
; CHECK:       vector.body:
; CHECK:         [[CALL_LANE_0:%.*]] = tail call { float, float } @foo(float {{%.*}})
; CHECK:         [[CALL_LANE_1:%.*]] = tail call { float, float } @foo(float {{%.*}})
;                // Lane 0
; CHECK:         [[A_0:%.*]] = extractvalue { float, float } [[CALL_LANE_0]], 0
; CHECK:         [[VEC_A_0:%.*]] = insertelement <2 x float> poison, float [[A_0]], i32 0
; CHECK:         [[WIDE_A_0:%.*]] = insertvalue { <2 x float>, <2 x float> } poison, <2 x float> [[VEC_A_0]], 0
; CHECK:         [[B_0:%.*]] = extractvalue { float, float } [[CALL_LANE_0]], 1
; CHECK:         [[UNDEF_B_0:%.*]] = extractvalue { <2 x float>, <2 x float> } [[WIDE_A_0]], 1
; CHECK:         [[VEC_B_0:%.*]] = insertelement <2 x float> [[UNDEF_B_0]], float [[B_0]], i32 0
; CHECK:         [[WIDE_0:%.*]] = insertvalue { <2 x float>, <2 x float> } [[WIDE_A_0]], <2 x float> [[VEC_B_0]], 1
;                // Lane 1
; CHECK:         [[A_1:%.*]] = extractvalue { float, float } [[CALL_LANE_1]], 0
; CHECK:         [[VEC_A_0_EXT:%.*]] = extractvalue { <2 x float>, <2 x float> } [[WIDE_0]], 0
; CHECK:         [[VEC_A:%.*]] = insertelement <2 x float> [[VEC_A_0_EXT]], float [[A_1]], i32 1
; CHECK:         [[WIDE_A:%.*]] = insertvalue { <2 x float>, <2 x float> } [[WIDE_0]], <2 x float> [[VEC_A]], 0
; CHECK:         [[B_1:%.*]] = extractvalue { float, float } [[CALL_LANE_1]], 1
; CHECK:         [[VEC_B_0_EXT:%.*]] = extractvalue { <2 x float>, <2 x float> } [[WIDE_A]], 1
; CHECK:         [[VEC_B:%.*]] = insertelement <2 x float> [[VEC_B_0_EXT]], float [[B_1]], i32 1
; CHECK:         [[WIDE:%.*]] = insertvalue { <2 x float>, <2 x float> } [[WIDE_A]], <2 x float> [[VEC_B]], 1
;                // Store wide values:
; CHECK:         [[VEC_A_EXT:%.*]] = extractvalue { <2 x float>, <2 x float> } [[WIDE]], 0
; CHECK:         [[VEC_B_EXT:%.*]] = extractvalue { <2 x float>, <2 x float> } [[WIDE]], 1
; CHECK:         store <2 x float> [[VEC_A_EXT]], ptr {{%.*}}, align 4
; CHECK:         store <2 x float> [[VEC_B_EXT]], ptr {{%.*}}, align 4
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %iv
  %in_val = load float, ptr %arrayidx, align 4
  ; #3 does not have a fixed-size vector mapping (so replication is used)
  %call = tail call { float, float } @foo(float %in_val) #3
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

; CHECK-REMARKS: remark: {{.*}} vectorized loop
define void @struct_return_f32_widen_rt_checks(ptr %in, ptr writeonly %out_a, ptr writeonly %out_b) {
; CHECK-LABEL: define void @struct_return_f32_widen_rt_checks
; CHECK-SAME:  (ptr [[IN:%.*]], ptr writeonly [[OUT_A:%.*]], ptr writeonly [[OUT_B:%.*]])
; CHECK:       entry:
; CHECK:         br i1 false, label %scalar.ph, label %vector.memcheck
; CHECK:       vector.memcheck:
; CHECK:       vector.body:
; CHECK:         call { <2 x float>, <2 x float> } @fixed_vec_foo(<2 x float> [[WIDE_LOAD:%.*]])
; CHECK:       for.body:
; CHECK          call { float, float } @foo(float [[LOAD:%.*]])
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

; TODO: Allow mixed-struct type vectorization and mark overflow intrinsics as trivially vectorizable.
; CHECK-REMARKS:         remark: {{.*}} loop not vectorized: call instruction cannot be vectorized
define void @test_overflow_intrinsic(ptr noalias readonly %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: define void @test_overflow_intrinsic
; CHECK-NOT:   vector.body:
; CHECK-NOT:   @llvm.sadd.with.overflow.v{{.+}}i32
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %iv
  %in_val = load i32, ptr %arrayidx, align 4
  %call = tail call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %in_val, i32 %in_val)
  %extract_ret = extractvalue { i32, i1 } %call, 0
  %extract_overflow = extractvalue { i32, i1 } %call, 1
  %zext_overflow = zext i1 %extract_overflow to i8
  %arrayidx2 = getelementptr inbounds i32, ptr %out_a, i64 %iv
  store i32 %extract_ret, ptr %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds i8, ptr %out_b, i64 %iv
  store i8 %zext_overflow, ptr %arrayidx4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

; CHECK-REMARKS: remark: {{.*}} vectorized loop
define void @struct_return_i32_three_results_widen(ptr noalias %in, ptr noalias writeonly %out_a) {
; CHECK-LABEL: define void @struct_return_i32_three_results_widen
; CHECK:   vector.body:
; CHECK:     call { <2 x i32>, <2 x i32>, <2 x i32> } @fixed_vec_qux(<2 x i32> [[WIDE_LOAD:%.*]])
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %in, i64 %iv
  %in_val = load i32, ptr %arrayidx, align 4
  %call = tail call { i32, i32, i32 } @qux(i32 %in_val) #5
  %extract_a = extractvalue { i32, i32, i32 } %call, 0
  %arrayidx2 = getelementptr inbounds i32, ptr %out_a, i64 %iv
  store i32 %extract_a, ptr %arrayidx2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

; Test crafted to exercise computePredInstDiscount with struct results
; (mainly it does not crash).
; CHECK-REMARKS: remark: {{.*}} vectorized loop
define void @scalarized_predicated_struct_return(ptr %a) {
; CHECK-LABEL: define void @scalarized_predicated_struct_return
; CHECK:  vector.body:
; CHECK:  pred.store.if:
; CHECK:     tail call { i64, i64 } @bar_i64(i64 %5)
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %in_val = load i64, ptr %arrayidx, align 8
  %sgt_zero = icmp sgt i64 %in_val, 0
  br i1 %sgt_zero, label %if.then, label %for.inc

if.then:
  %call = tail call { i64, i64 } @bar_i64(i64 %in_val) #6
  %extract_a = extractvalue { i64, i64 } %call, 0
  %div = udiv i64 %extract_a, %in_val
  store i64 %div, ptr %arrayidx, align 8
  br label %for.inc

for.inc:
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

; Negative test. Widening structs of vectors is not supported.
; CHECK-REMARKS-COUNT: remark: {{.*}} loop not vectorized: instruction return type cannot be vectorized
define void @negative_struct_of_vectors(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: define void @negative_struct_of_vectors
; CHECK-NOT:   vector.body:
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %iv
  %in_val = load <1 x float>, ptr %arrayidx, align 4
  %call = tail call { <1 x float>, <1 x float> } @foo(<1 x float> %in_val) #0
  %extract_a = extractvalue { <1 x float>, <1 x float> } %call, 0
  %extract_b = extractvalue { <1 x float>, <1 x float> } %call, 1
  %arrayidx2 = getelementptr inbounds float, ptr %out_a, i64 %iv
  store <1 x float> %extract_a, ptr %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds float, ptr %out_b, i64 %iv
  store <1 x float> %extract_b, ptr %arrayidx4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

; Negative test. Widening structs with mixed element types is not supported.
; CHECK-REMARKS-COUNT: remark: {{.*}} loop not vectorized: instruction return type cannot be vectorized
define void @negative_mixed_element_type_struct_return(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: define void @negative_mixed_element_type_struct_return
; CHECK-NOT:   vector.body:
; CHECK-NOT:   call {{.*}} @fixed_vec_baz
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %iv
  %in_val = load float, ptr %arrayidx, align 4
  %call = tail call { float, i32 } @baz(float %in_val) #2
  %extract_a = extractvalue { float, i32 } %call, 0
  %extract_b = extractvalue { float, i32 } %call, 1
  %arrayidx2 = getelementptr inbounds float, ptr %out_a, i64 %iv
  store float %extract_a, ptr %arrayidx2, align 4
  %arrayidx4 = getelementptr inbounds i32, ptr %out_b, i64 %iv
  store i32 %extract_b, ptr %arrayidx4, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

%named_struct = type { double, double }

; Negative test. Widening non-literal structs is not supported.
; CHECK-REMARKS-COUNT: remark: {{.*}} loop not vectorized: instruction return type cannot be vectorized
define void @negative_named_struct_return(ptr noalias readonly %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: define void @negative_named_struct_return
; CHECK-NOT:   vector.body:
; CHECK-NOT:   call {{.*}} @fixed_vec_bar
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %in, i64 %iv
  %in_val = load double, ptr %arrayidx, align 8
  %call = tail call %named_struct @bar_named(double %in_val) #4
  %extract_a = extractvalue %named_struct %call, 0
  %extract_b = extractvalue %named_struct %call, 1
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

; Negative test. Nested homogeneous structs are not supported.
; CHECK-REMARKS-COUNT: remark: {{.*}} loop not vectorized: instruction return type cannot be vectorized
define void @negative_nested_struct(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: define void @negative_nested_struct
; CHECK-NOT:   vector.body:
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %iv
  %in_val = load float, ptr %arrayidx, align 4
  %call = tail call { { float, float } } @foo_nested_struct(float %in_val) #0
  %extract_inner = extractvalue { { float, float } } %call, 0
  %extract_a = extractvalue { float, float } %extract_inner, 0
  %extract_b = extractvalue { float, float } %extract_inner, 1
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

; Negative test. The second element of the struct cannot be widened.
; CHECK-REMARKS-COUNT: remark: {{.*}} loop not vectorized: instruction return type cannot be vectorized
define void @negative_non_widenable_element(ptr noalias %in, ptr noalias writeonly %out_a) {
; CHECK-LABEL: define void @negative_non_widenable_element
; CHECK-NOT:   vector.body:
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %iv
  %in_val = load float, ptr %arrayidx, align 4
  %call = tail call { float, [1 x float] } @foo_one_non_widenable_element(float %in_val) #0
  %extract_a = extractvalue { float, [1 x float] } %call, 0
  %arrayidx2 = getelementptr inbounds float, ptr %out_a, i64 %iv
  store float %extract_a, ptr %arrayidx2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

; Negative test. Homogeneous structs of arrays are not supported.
; CHECK-REMARKS-COUNT: remark: {{.*}} loop not vectorized: instruction return type cannot be vectorized
define void @negative_struct_array_elements(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: define void @negative_struct_array_elements
; CHECK-NOT:   vector.body:
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %in, i64 %iv
  %in_val = load float, ptr %arrayidx, align 4
  %call = tail call { [2 x float] } @foo_arrays(float %in_val) #0
  %extract_inner = extractvalue { [2 x float] } %call, 0
  %extract_a = extractvalue [2 x float] %extract_inner, 0
  %extract_b = extractvalue [2 x float] %extract_inner, 1
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

; Negative test. Widening struct loads is not supported.
; CHECK-REMARKS: remark: {{.*}} loop not vectorized: instruction return type cannot be vectorized
define void @negative_struct_load(ptr noalias %in, ptr noalias writeonly %out_a, ptr noalias writeonly %out_b) {
; CHECK-LABEL: define void @negative_struct_load
; CHECK-NOT:   vector.body:
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds { float, float }, ptr %in, i64 %iv
  %call = load { float, float }, ptr %arrayidx, align 8
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

; Negative test. Widening struct stores is not supported.
; CHECK-REMARKS: remark: {{.*}} loop not vectorized: instruction return type cannot be vectorized
define void @negative_struct_return_store_struct(ptr noalias %in, ptr noalias writeonly %out) {
; CHECK-LABEL: define void @negative_struct_return_store_struct
; CHECK-NOT:   vector.body:
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds { float, float }, ptr %in, i64 %iv
  %in_val = load float, ptr %arrayidx, align 4
  %call = tail call { float, float } @foo(float %in_val) #0
  %out_ptr = getelementptr inbounds { float, float }, ptr %out, i64 %iv
  store { float, float } %call, ptr %out_ptr, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

declare { float, float } @foo(float)
declare { double, double } @bar(double)
declare { float, i32 } @baz(float)
declare %named_struct @bar_named(double)
declare { { float, float } } @foo_nested_struct(float)
declare { [2 x float] } @foo_arrays(float)
declare { float, [1 x float] } @foo_one_non_widenable_element(float)
declare { <1 x float>, <1 x float> } @foo_vectors(<1 x float>)
declare { i32, i32, i32 } @qux(i32)
declare { i64, i64 } @bar_i64(i64)

declare { <2 x float>, <2 x float> } @fixed_vec_foo(<2 x float>)
declare { <2 x double>, <2 x double> } @fixed_vec_bar(<2 x double>)
declare { <2 x float>, <2 x i32> } @fixed_vec_baz(<2 x float>)
declare { <2 x i32>, <2 x i32>, <2 x i32> } @fixed_vec_qux(<2 x i32>)
declare { <vscale x 4 x float>, <vscale x 4 x float> } @scalable_vec_masked_foo(<vscale x 4 x float>, <vscale x 4 x i1>)
declare { <vscale x 4 x i64>, <vscale x 4 x i64> } @scalable_vec_masked_bar_i64(<vscale x 4 x i64>, <vscale x 4 x i1>)

attributes #0 = { nounwind "vector-function-abi-variant"="_ZGVnN2v_foo(fixed_vec_foo)" }
attributes #1 = { nounwind "vector-function-abi-variant"="_ZGVnN2v_bar(fixed_vec_bar)" }
attributes #2 = { nounwind "vector-function-abi-variant"="_ZGVnN2v_baz(fixed_vec_baz)" }
attributes #3 = { nounwind "vector-function-abi-variant"="_ZGVsMxv_foo(scalable_vec_masked_foo)" }
attributes #4 = { nounwind "vector-function-abi-variant"="_ZGVnN2v_bar_named(fixed_vec_bar)" }
attributes #5 = { nounwind "vector-function-abi-variant"="_ZGVnN2v_qux(fixed_vec_qux)" }
attributes #6 = { nounwind "vector-function-abi-variant"="_ZGVsMxv_bar_i64(scalable_vec_masked_bar_i64)" }
