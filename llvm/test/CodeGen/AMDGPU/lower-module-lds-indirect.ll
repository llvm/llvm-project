; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

; CHECK: %llvm.amdgcn.module.lds.t = type { double, float }

; CHECK: @function_indirect = addrspace(1) global float* addrspacecast (float addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 1) to float*), align 8

; CHECK: @kernel_indirect = addrspace(1) global double* addrspacecast (double addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 0) to double*), align 8

; CHECK: @llvm.amdgcn.module.lds = internal addrspace(3) global %llvm.amdgcn.module.lds.t undef, align 8

@function_target = addrspace(3) global float undef, align 4
@function_indirect = addrspace(1) global float* addrspacecast (float addrspace(3)* @function_target to float*), align 8

@kernel_target = addrspace(3) global double undef, align 8
@kernel_indirect = addrspace(1) global double* addrspacecast (double addrspace(3)* @kernel_target to double*), align 8

; CHECK-LABEL: @function(float %x)
; CHECK: %0 = load float*, float* addrspace(1)* @function_indirect, align 8
define void @function(float %x) local_unnamed_addr #5 {
entry:
  %0 = load float*, float* addrspace(1)* @function_indirect, align 8
  store float %x, float* %0, align 4
  ret void
}

; CHECK-LABEL: @kernel(double %x)
; CHECK: call void @llvm.donothing() [ "ExplicitUse"(%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds) ]
; CHECK: %0 = load double*, double* addrspace(1)* @kernel_indirect, align 8
define amdgpu_kernel void @kernel(double %x) local_unnamed_addr #5 {
entry:
  %0 = load double*, double* addrspace(1)* @kernel_indirect, align 8
  store double %x, double* %0, align 8
  ret void
}




