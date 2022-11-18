; RUN: opt -data-layout=A5 -S -mtriple=amdgcn-unknown-unknown -passes=amdgpu-promote-alloca < %s | FileCheck -check-prefix=IR %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=ASM %s


@global_array0 = internal unnamed_addr addrspace(3) global [750 x [10 x i32]] undef, align 4
@global_array1 = internal unnamed_addr addrspace(3) global [750 x [10 x i32]] undef, align 4

; IR-LABEL: define amdgpu_kernel void @promote_alloca_size_256(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) {
; IR: alloca [10 x i32]
; ASM-LABEL: {{^}}promote_alloca_size_256:
; ASM: .amdgpu_lds llvm.amdgcn.kernel.promote_alloca_size_256.lds, 60000, 16
; ASM-NOT: .amdgpu_lds

define amdgpu_kernel void @promote_alloca_size_256(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) {
entry:
  %stack = alloca [10 x i32], align 4, addrspace(5)
  %tmp = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [10 x i32], [10 x i32] addrspace(5)* %stack, i32 0, i32 %tmp
  store i32 4, i32 addrspace(5)* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 1
  %tmp1 = load i32, i32 addrspace(1)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [10 x i32], [10 x i32] addrspace(5)* %stack, i32 0, i32 %tmp1
  store i32 5, i32 addrspace(5)* %arrayidx3, align 4
  %arrayidx10 = getelementptr inbounds [10 x i32], [10 x i32] addrspace(5)* %stack, i32 0, i32 0
  %tmp2 = load i32, i32 addrspace(5)* %arrayidx10, align 4
  store i32 %tmp2, i32 addrspace(1)* %out, align 4
  %arrayidx12 = getelementptr inbounds [10 x i32], [10 x i32] addrspace(5)* %stack, i32 0, i32 1
  %tmp3 = load i32, i32 addrspace(5)* %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 1
  store i32 %tmp3, i32 addrspace(1)* %arrayidx13
  %v0 = getelementptr inbounds [750 x [10 x i32]], [750 x [10 x i32]] addrspace(3)* @global_array0, i32 0, i32 0, i32 0
  store i32 %tmp3, i32 addrspace(3)* %v0
  %v1 = getelementptr inbounds [750 x [10 x i32]], [750 x [10 x i32]] addrspace(3)* @global_array1, i32 0, i32 0, i32 0
  store i32 %tmp3, i32 addrspace(3)* %v1
  ret void
}
