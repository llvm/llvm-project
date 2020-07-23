; RUN: opt -S -disable-promote-alloca-to-vector -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-promote-alloca < %s | FileCheck -check-prefix=IR %s
; RUN: llc -disable-promote-alloca-to-vector -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefix=ASM %s

target datalayout = "A5"

@all_lds = internal unnamed_addr addrspace(3) global [16384 x i32] undef, align 4

; This function cannot promote to using LDS because of the size of the
; constant expression use in the function, which was previously not
; detected.
; IR-LABEL: @constant_expression_uses_lds(
; IR: alloca

; ASM-LABEL: constant_expression_uses_lds:
; ASM: .group_segment_fixed_size: 65536
define amdgpu_kernel void @constant_expression_uses_lds(i32 addrspace(1)* nocapture %out, i32 %idx) #0 {
entry:
  %stack = alloca [4 x i32], align 4, addrspace(5)
  %gep0 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(5)* %stack, i32 0, i32 0
  %gep1 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(5)* %stack, i32 0, i32 1
  %gep2 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(5)* %stack, i32 0, i32 2
  %gep3 = getelementptr inbounds [4 x i32], [4 x i32] addrspace(5)* %stack, i32 0, i32 3
  store i32 9, i32 addrspace(5)* %gep0
  store i32 10, i32 addrspace(5)* %gep1
  store i32 99, i32 addrspace(5)* %gep2
  store i32 43, i32 addrspace(5)* %gep3
  %arrayidx = getelementptr inbounds [4 x i32], [4 x i32] addrspace(5)* %stack, i32 0, i32 %idx
  %load = load i32, i32 addrspace(5)* %arrayidx, align 4
  store i32 %load, i32 addrspace(1)* %out

  store volatile i32 ptrtoint ([16384 x i32] addrspace(3)* @all_lds to i32), i32 addrspace(1)* undef
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="1,5" }
