; RUN: opt -amdgpu-vector-idiom-enable -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-vector-idiom -S %s | FileCheck %s
;
; Simple testcase for accumulator vectorization

define amdgpu_kernel void @simple_accumulator(i32 %idx) {
; CHECK-LABEL: define amdgpu_kernel void @simple_accumulator(
; CHECK-SAME: i32 [[IDX:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[ACCUMULATORS:%.*]] = alloca [4 x float], align 16, addrspace(5)
; CHECK:         [[VEC_LOAD:%.*]] = load <4 x float>, ptr addrspace(5) [[ACCUMULATORS]], align 16
; CHECK-NOT:     load float, ptr addrspace(5)
; CHECK:         ret void
;
entry:
  %accumulators = alloca [4 x float], align 16, addrspace(5)
  %load0 = load float, ptr addrspace(5) %accumulators, align 16
  %gep1 = getelementptr inbounds [4 x float], ptr addrspace(5) %accumulators, i32 0, i32 1
  %load1 = load float, ptr addrspace(5) %gep1, align 4
  %gep2 = getelementptr inbounds [4 x float], ptr addrspace(5) %accumulators, i32 0, i32 2
  %load2 = load float, ptr addrspace(5) %gep2, align 8
  %gep3 = getelementptr inbounds [4 x float], ptr addrspace(5) %accumulators, i32 0, i32 3
  %load3 = load float, ptr addrspace(5) %gep3, align 4
  ret void
}