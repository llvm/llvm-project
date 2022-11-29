; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=infer-address-spaces %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes=infer-address-spaces %s | FileCheck %s
; Ports of most of test/CodeGen/NVPTX/access-non-generic.ll

@scalar = internal addrspace(3) global float 0.0, align 4
@array = internal addrspace(3) global [10 x float] zeroinitializer, align 4

; CHECK-LABEL: @load_store_lds_f32(
; CHECK: %tmp = load float, ptr addrspace(3) @scalar, align 4
; CHECK: call void @use(float %tmp)
; CHECK: store float %v, ptr addrspace(3) @scalar, align 4
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: %tmp2 = load float, ptr addrspace(3) @scalar, align 4
; CHECK: call void @use(float %tmp2)
; CHECK: store float %v, ptr addrspace(3) @scalar, align 4
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: %tmp3 = load float, ptr addrspace(3) getelementptr inbounds ([10 x float], ptr addrspace(3) @array, i32 0, i32 5), align 4
; CHECK: call void @use(float %tmp3)
; CHECK: store float %v, ptr addrspace(3) getelementptr inbounds ([10 x float], ptr addrspace(3) @array, i32 0, i32 5), align 4
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: %tmp4 = getelementptr inbounds [10 x float], ptr addrspace(3) @array, i32 0, i32 5
; CHECK: %tmp5 = load float, ptr addrspace(3) %tmp4, align 4
; CHECK: call void @use(float %tmp5)
; CHECK: store float %v, ptr addrspace(3) %tmp4, align 4
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: %tmp7 = getelementptr inbounds [10 x float], ptr addrspace(3) @array, i32 0, i32 %i
; CHECK: %tmp8 = load float, ptr addrspace(3) %tmp7, align 4
; CHECK: call void @use(float %tmp8)
; CHECK: store float %v, ptr addrspace(3) %tmp7, align 4
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: ret void
define amdgpu_kernel void @load_store_lds_f32(i32 %i, float %v) #0 {
bb:
  %tmp = load float, ptr addrspacecast (ptr addrspace(3) @scalar to ptr), align 4
  call void @use(float %tmp)
  store float %v, ptr addrspacecast (ptr addrspace(3) @scalar to ptr), align 4
  call void @llvm.amdgcn.s.barrier()
  %tmp1 = addrspacecast ptr addrspace(3) @scalar to ptr
  %tmp2 = load float, ptr %tmp1, align 4
  call void @use(float %tmp2)
  store float %v, ptr %tmp1, align 4
  call void @llvm.amdgcn.s.barrier()
  %tmp3 = load float, ptr getelementptr inbounds ([10 x float], ptr addrspacecast (ptr addrspace(3) @array to ptr), i32 0, i32 5), align 4
  call void @use(float %tmp3)
  store float %v, ptr getelementptr inbounds ([10 x float], ptr addrspacecast (ptr addrspace(3) @array to ptr), i32 0, i32 5), align 4
  call void @llvm.amdgcn.s.barrier()
  %tmp4 = getelementptr inbounds [10 x float], ptr addrspacecast (ptr addrspace(3) @array to ptr), i32 0, i32 5
  %tmp5 = load float, ptr %tmp4, align 4
  call void @use(float %tmp5)
  store float %v, ptr %tmp4, align 4
  call void @llvm.amdgcn.s.barrier()
  %tmp6 = addrspacecast ptr addrspace(3) @array to ptr
  %tmp7 = getelementptr inbounds [10 x float], ptr %tmp6, i32 0, i32 %i
  %tmp8 = load float, ptr %tmp7, align 4
  call void @use(float %tmp8)
  store float %v, ptr %tmp7, align 4
  call void @llvm.amdgcn.s.barrier()
  ret void
}

; CHECK-LABEL: @constexpr_load_int_from_float_lds(
; CHECK: %tmp = load i32, ptr addrspace(3) @scalar, align 4
define i32 @constexpr_load_int_from_float_lds() #0 {
bb:
  %tmp = load i32, ptr addrspacecast (ptr addrspace(3) @scalar to ptr), align 4
  ret i32 %tmp
}

; CHECK-LABEL: @load_int_from_global_float(
; CHECK: %tmp1 = getelementptr float, ptr addrspace(1) %input, i32 %i
; CHECK: %tmp2 = getelementptr float, ptr addrspace(1) %tmp1, i32 %j
; CHECK: %tmp4 = load i32, ptr addrspace(1) %tmp2
; CHECK: ret i32 %tmp4
define i32 @load_int_from_global_float(ptr addrspace(1) %input, i32 %i, i32 %j) #0 {
bb:
  %tmp = addrspacecast ptr addrspace(1) %input to ptr
  %tmp1 = getelementptr float, ptr %tmp, i32 %i
  %tmp2 = getelementptr float, ptr %tmp1, i32 %j
  %tmp4 = load i32, ptr %tmp2
  ret i32 %tmp4
}

; CHECK-LABEL: @nested_const_expr(
; CHECK: store i32 1, ptr addrspace(3) getelementptr inbounds ([10 x float], ptr addrspace(3) @array, i64 0, i64 1), align 4
define amdgpu_kernel void @nested_const_expr() #0 {
  store i32 1, ptr bitcast (ptr getelementptr ([10 x float], ptr addrspacecast (ptr addrspace(3) @array to ptr), i64 0, i64 1) to ptr), align 4

  ret void
}

; CHECK-LABEL: @rauw(
; CHECK: %addr = getelementptr float, ptr addrspace(1) %input, i64 10
; CHECK-NEXT: %v = load float, ptr addrspace(1) %addr
; CHECK-NEXT: store float %v, ptr addrspace(1) %addr
; CHECK-NEXT: ret void
define amdgpu_kernel void @rauw(ptr addrspace(1) %input) #0 {
bb:
  %generic_input = addrspacecast ptr addrspace(1) %input to ptr
  %addr = getelementptr float, ptr %generic_input, i64 10
  %v = load float, ptr %addr
  store float %v, ptr %addr
  ret void
}

; FIXME: Should be able to eliminate the cast inside the loop
; CHECK-LABEL: @loop(

; CHECK: %end = getelementptr float, ptr addrspace(3) @array, i64 10
; CHECK: br label %loop

; CHECK: loop:                                             ; preds = %loop, %entry
; CHECK: %i = phi ptr addrspace(3) [ @array, %entry ], [ %i2, %loop ]
; CHECK: %v = load float, ptr addrspace(3) %i
; CHECK: call void @use(float %v)
; CHECK: %i2 = getelementptr float, ptr addrspace(3) %i, i64 1
; CHECK: %exit_cond = icmp eq ptr addrspace(3) %i2, %end

; CHECK: br i1 %exit_cond, label %exit, label %loop
define amdgpu_kernel void @loop() #0 {
entry:
  %p = addrspacecast ptr addrspace(3) @array to ptr
  %end = getelementptr float, ptr %p, i64 10
  br label %loop

loop:                                             ; preds = %loop, %entry
  %i = phi ptr [ %p, %entry ], [ %i2, %loop ]
  %v = load float, ptr %i
  call void @use(float %v)
  %i2 = getelementptr float, ptr %i, i64 1
  %exit_cond = icmp eq ptr %i2, %end
  br i1 %exit_cond, label %exit, label %loop

exit:                                             ; preds = %loop
  ret void
}

@generic_end = external addrspace(1) global ptr

; CHECK-LABEL: @loop_with_generic_bound(
; CHECK: %end = load ptr, ptr addrspace(1) @generic_end
; CHECK: br label %loop

; CHECK: loop:
; CHECK: %i = phi ptr addrspace(3) [ @array, %entry ], [ %i2, %loop ]
; CHECK: %v = load float, ptr addrspace(3) %i
; CHECK: call void @use(float %v)
; CHECK: %i2 = getelementptr float, ptr addrspace(3) %i, i64 1
; CHECK: %0 = addrspacecast ptr addrspace(3) %i2 to ptr
; CHECK: %exit_cond = icmp eq ptr %0, %end
; CHECK: br i1 %exit_cond, label %exit, label %loop
define amdgpu_kernel void @loop_with_generic_bound() #0 {
entry:
  %p = addrspacecast ptr addrspace(3) @array to ptr
  %end = load ptr, ptr addrspace(1) @generic_end
  br label %loop

loop:                                             ; preds = %loop, %entry
  %i = phi ptr [ %p, %entry ], [ %i2, %loop ]
  %v = load float, ptr %i
  call void @use(float %v)
  %i2 = getelementptr float, ptr %i, i64 1
  %exit_cond = icmp eq ptr %i2, %end
  br i1 %exit_cond, label %exit, label %loop

exit:                                             ; preds = %loop
  ret void
}

; CHECK-LABEL: @select_bug(
; CHECK: %add.ptr157 = getelementptr inbounds i64, ptr undef, i64 select (i1 icmp ne (ptr inttoptr (i64 4873 to ptr), ptr null), i64 73, i64 93)
; CHECK: %cmp169 = icmp uge ptr undef, %add.ptr157
define void @select_bug() #0 {
  %add.ptr157 = getelementptr inbounds i64, ptr undef, i64 select (i1 icmp ne (ptr inttoptr (i64 4873 to ptr), ptr null), i64 73, i64 93)
  %cmp169 = icmp uge ptr undef, %add.ptr157
  unreachable
}

declare void @llvm.amdgcn.s.barrier() #1
declare void @use(float) #0

attributes #0 = { nounwind }
attributes #1 = { convergent nounwind }
