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

; OPT-LABEL: @promote_to_vector_call_c(
; OPT-NOT: alloca
; OPT: extractelement <2 x i32> %{{[0-9]+}}, i32 %in
; ASM-NOT: LDSByteSize
define void @promote_to_vector_call_c(i32 addrspace(1)* %out, i32 %in) #0 {
entry:
  %tmp = alloca [2 x i32]
  %tmp1 = getelementptr [2 x i32], [2 x i32]* %tmp, i32 0, i32 0
  %tmp2 = getelementptr [2 x i32], [2 x i32]* %tmp, i32 0, i32 1
  store i32 0, i32* %tmp1
  store i32 1, i32* %tmp2
  %tmp3 = getelementptr [2 x i32], [2 x i32]* %tmp, i32 0, i32 %in
  %tmp4 = load i32, i32* %tmp3
  %tmp5 = load volatile i32, i32 addrspace(1)* undef
  %tmp6 = add i32 %tmp4, %tmp5
  store i32 %tmp6, i32 addrspace(1)* %out
  ret void
}

; OPT-LABEL: @no_promote_to_lds_c(
; OPT: alloca
; ASM-NOT: LDSByteSize
define void @no_promote_to_lds(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in) #0 {
entry:
  %stack = alloca [5 x i32], align 4
  %0 = load i32, i32 addrspace(1)* %in, align 4
  %arrayidx1 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 %0
  store i32 4, i32* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 1
  %1 = load i32, i32 addrspace(1)* %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 %1
  store i32 5, i32* %arrayidx3, align 4
  %arrayidx10 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 0
  %2 = load i32, i32* %arrayidx10, align 4
  store i32 %2, i32 addrspace(1)* %out, align 4
  %arrayidx12 = getelementptr inbounds [5 x i32], [5 x i32]* %stack, i32 0, i32 1
  %3 = load i32, i32* %arrayidx12
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 1
  store i32 %3, i32 addrspace(1)* %arrayidx13
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind "amdgpu-max-work-group-size"="64" }
attributes #1 = { nounwind readnone }
