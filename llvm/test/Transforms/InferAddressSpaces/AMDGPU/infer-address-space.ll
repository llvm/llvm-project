; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -infer-address-spaces %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=infer-address-spaces %s | FileCheck %s
; Ports of most of test/CodeGen/NVPTX/access-non-generic.ll

@scalar = internal addrspace(3) global float 0.0, align 4
@array = internal addrspace(3) global [10 x float] zeroinitializer, align 4

; CHECK-LABEL: @load_store_lds_f32(
; CHECK: %tmp = load float, float addrspace(3)* @scalar, align 4
; CHECK: call void @use(float %tmp)
; CHECK: store float %v, float addrspace(3)* @scalar, align 4
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: %tmp2 = load float, float addrspace(3)* @scalar, align 4
; CHECK: call void @use(float %tmp2)
; CHECK: store float %v, float addrspace(3)* @scalar, align 4
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: %tmp3 = load float, float addrspace(3)* getelementptr inbounds ([10 x float], [10 x float] addrspace(3)* @array, i32 0, i32 5), align 4
; CHECK: call void @use(float %tmp3)
; CHECK: store float %v, float addrspace(3)* getelementptr inbounds ([10 x float], [10 x float] addrspace(3)* @array, i32 0, i32 5), align 4
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: %tmp4 = getelementptr inbounds [10 x float], [10 x float] addrspace(3)* @array, i32 0, i32 5
; CHECK: %tmp5 = load float, float addrspace(3)* %tmp4, align 4
; CHECK: call void @use(float %tmp5)
; CHECK: store float %v, float addrspace(3)* %tmp4, align 4
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: %tmp7 = getelementptr inbounds [10 x float], [10 x float] addrspace(3)* @array, i32 0, i32 %i
; CHECK: %tmp8 = load float, float addrspace(3)* %tmp7, align 4
; CHECK: call void @use(float %tmp8)
; CHECK: store float %v, float addrspace(3)* %tmp7, align 4
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: ret void
define amdgpu_kernel void @load_store_lds_f32(i32 %i, float %v) #0 {
bb:
  %tmp = load float, float* addrspacecast (float addrspace(3)* @scalar to float*), align 4
  call void @use(float %tmp)
  store float %v, float* addrspacecast (float addrspace(3)* @scalar to float*), align 4
  call void @llvm.amdgcn.s.barrier()
  %tmp1 = addrspacecast float addrspace(3)* @scalar to float*
  %tmp2 = load float, float* %tmp1, align 4
  call void @use(float %tmp2)
  store float %v, float* %tmp1, align 4
  call void @llvm.amdgcn.s.barrier()
  %tmp3 = load float, float* getelementptr inbounds ([10 x float], [10 x float]* addrspacecast ([10 x float] addrspace(3)* @array to [10 x float]*), i32 0, i32 5), align 4
  call void @use(float %tmp3)
  store float %v, float* getelementptr inbounds ([10 x float], [10 x float]* addrspacecast ([10 x float] addrspace(3)* @array to [10 x float]*), i32 0, i32 5), align 4
  call void @llvm.amdgcn.s.barrier()
  %tmp4 = getelementptr inbounds [10 x float], [10 x float]* addrspacecast ([10 x float] addrspace(3)* @array to [10 x float]*), i32 0, i32 5
  %tmp5 = load float, float* %tmp4, align 4
  call void @use(float %tmp5)
  store float %v, float* %tmp4, align 4
  call void @llvm.amdgcn.s.barrier()
  %tmp6 = addrspacecast [10 x float] addrspace(3)* @array to [10 x float]*
  %tmp7 = getelementptr inbounds [10 x float], [10 x float]* %tmp6, i32 0, i32 %i
  %tmp8 = load float, float* %tmp7, align 4
  call void @use(float %tmp8)
  store float %v, float* %tmp7, align 4
  call void @llvm.amdgcn.s.barrier()
  ret void
}

; CHECK-LABEL: @constexpr_load_int_from_float_lds(
; CHECK: %tmp = load i32, i32 addrspace(3)* bitcast (float addrspace(3)* @scalar to i32 addrspace(3)*), align 4
define i32 @constexpr_load_int_from_float_lds() #0 {
bb:
  %tmp = load i32, i32* addrspacecast (i32 addrspace(3)* bitcast (float addrspace(3)* @scalar to i32 addrspace(3)*) to i32*), align 4
  ret i32 %tmp
}

; CHECK-LABEL: @load_int_from_global_float(
; CHECK: %tmp1 = getelementptr float, float addrspace(1)* %input, i32 %i
; CHECK: %tmp2 = getelementptr float, float addrspace(1)* %tmp1, i32 %j
; CHECK: %tmp3 = bitcast float addrspace(1)* %tmp2 to i32 addrspace(1)*
; CHECK: %tmp4 = load i32, i32 addrspace(1)* %tmp3
; CHECK: ret i32 %tmp4
define i32 @load_int_from_global_float(float addrspace(1)* %input, i32 %i, i32 %j) #0 {
bb:
  %tmp = addrspacecast float addrspace(1)* %input to float*
  %tmp1 = getelementptr float, float* %tmp, i32 %i
  %tmp2 = getelementptr float, float* %tmp1, i32 %j
  %tmp3 = bitcast float* %tmp2 to i32*
  %tmp4 = load i32, i32* %tmp3
  ret i32 %tmp4
}

; CHECK-LABEL: @nested_const_expr(
; CHECK: store i32 1, i32 addrspace(3)* bitcast (float addrspace(3)* getelementptr inbounds ([10 x float], [10 x float] addrspace(3)* @array, i64 0, i64 1) to i32 addrspace(3)*), align 4
define amdgpu_kernel void @nested_const_expr() #0 {
  store i32 1, i32* bitcast (float* getelementptr ([10 x float], [10 x float]* addrspacecast ([10 x float] addrspace(3)* @array to [10 x float]*), i64 0, i64 1) to i32*), align 4
  ret void
}

; CHECK-LABEL: @rauw(
; CHECK: %addr = getelementptr float, float addrspace(1)* %input, i64 10
; CHECK-NEXT: %v = load float, float addrspace(1)* %addr
; CHECK-NEXT: store float %v, float addrspace(1)* %addr
; CHECK-NEXT: ret void
define amdgpu_kernel void @rauw(float addrspace(1)* %input) #0 {
bb:
  %generic_input = addrspacecast float addrspace(1)* %input to float*
  %addr = getelementptr float, float* %generic_input, i64 10
  %v = load float, float* %addr
  store float %v, float* %addr
  ret void
}

; FIXME: Should be able to eliminate the cast inside the loop
; CHECK-LABEL: @loop(

; CHECK: %p = bitcast [10 x float] addrspace(3)* @array to float addrspace(3)*
; CHECK: %end = getelementptr float, float addrspace(3)* %p, i64 10
; CHECK: br label %loop

; CHECK: loop:                                             ; preds = %loop, %entry
; CHECK: %i = phi float addrspace(3)* [ %p, %entry ], [ %i2, %loop ]
; CHECK: %v = load float, float addrspace(3)* %i
; CHECK: call void @use(float %v)
; CHECK: %i2 = getelementptr float, float addrspace(3)* %i, i64 1
; CHECK: %exit_cond = icmp eq float addrspace(3)* %i2, %end

; CHECK: br i1 %exit_cond, label %exit, label %loop
define amdgpu_kernel void @loop() #0 {
entry:
  %p = addrspacecast [10 x float] addrspace(3)* @array to float*
  %end = getelementptr float, float* %p, i64 10
  br label %loop

loop:                                             ; preds = %loop, %entry
  %i = phi float* [ %p, %entry ], [ %i2, %loop ]
  %v = load float, float* %i
  call void @use(float %v)
  %i2 = getelementptr float, float* %i, i64 1
  %exit_cond = icmp eq float* %i2, %end
  br i1 %exit_cond, label %exit, label %loop

exit:                                             ; preds = %loop
  ret void
}

@generic_end = external addrspace(1) global float*

; CHECK-LABEL: @loop_with_generic_bound(
; CHECK: %p = bitcast [10 x float] addrspace(3)* @array to float addrspace(3)*
; CHECK: %end = load float*, float* addrspace(1)* @generic_end
; CHECK: br label %loop

; CHECK: loop:
; CHECK: %i = phi float addrspace(3)* [ %p, %entry ], [ %i2, %loop ]
; CHECK: %v = load float, float addrspace(3)* %i
; CHECK: call void @use(float %v)
; CHECK: %i2 = getelementptr float, float addrspace(3)* %i, i64 1
; CHECK: %0 = addrspacecast float addrspace(3)* %i2 to float*
; CHECK: %exit_cond = icmp eq float* %0, %end
; CHECK: br i1 %exit_cond, label %exit, label %loop
define amdgpu_kernel void @loop_with_generic_bound() #0 {
entry:
  %p = addrspacecast [10 x float] addrspace(3)* @array to float*
  %end = load float*, float* addrspace(1)* @generic_end
  br label %loop

loop:                                             ; preds = %loop, %entry
  %i = phi float* [ %p, %entry ], [ %i2, %loop ]
  %v = load float, float* %i
  call void @use(float %v)
  %i2 = getelementptr float, float* %i, i64 1
  %exit_cond = icmp eq float* %i2, %end
  br i1 %exit_cond, label %exit, label %loop

exit:                                             ; preds = %loop
  ret void
}

; CHECK-LABEL: @select_bug(
; CHECK: %add.ptr157 = getelementptr inbounds i64, i64* undef, i64 select (i1 icmp ne (i32* inttoptr (i64 4873 to i32*), i32* null), i64 73, i64 93)
; CHECK: %cmp169 = icmp uge i64* undef, %add.ptr157
define void @select_bug() #0 {
  %add.ptr157 = getelementptr inbounds i64, i64* undef, i64 select (i1 icmp ne (i32* inttoptr (i64 4873 to i32*), i32* null), i64 73, i64 93)
  %cmp169 = icmp uge i64* undef, %add.ptr157
  unreachable
}

declare void @llvm.amdgcn.s.barrier() #1
declare void @use(float) #0

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
