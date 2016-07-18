; RUN: opt -S -mtriple=amdgcn-unknown-unknown -amdgpu-promote-alloca < %s | FileCheck -check-prefix=IR %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=ASM %s

; IR-LABEL: define amdgpu_vs void @promote_alloca_shaders(i32 addrspace(1)* inreg %out, i32 addrspace(1)* inreg %in) #0 {
; IR: alloca [5 x i32]
; ASM-LABEL: {{^}}promote_alloca_shaders:
; ASM: ; LDSByteSize: 0 bytes/workgroup (compile time only)

define amdgpu_vs void @promote_alloca_shaders(i32 addrspace(1)* inreg %out, i32 addrspace(1)* inreg %in) #0 {
entry:
  %stack = alloca [5 x i32], align 4
  %tmp0 = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 %tmp0
  store i32 4, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 1
  %tmp1 = load i32, i32 addrspace(1)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 %tmp1
  store i32 5, i32* %arrayidx3, align 4
  %arrayidx4 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 0
  %tmp2 = load i32, i32* %arrayidx4, align 4
  store i32 %tmp2, i32 addrspace(1)* %out, align 4
  %arrayidx5 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 1
  %tmp3 = load i32, i32* %arrayidx5
  %arrayidx6 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 1
  store i32 %tmp3, i32 addrspace(1)* %arrayidx6
  ret void
}

attributes #0 = { nounwind "amdgpu-max-work-group-size"="64" }
