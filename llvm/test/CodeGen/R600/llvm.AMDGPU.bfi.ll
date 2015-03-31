; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i32 @llvm.AMDGPU.bfi(i32, i32, i32) nounwind readnone

; FUNC-LABEL: {{^}}bfi_arg_arg_arg:
; SI: v_bfi_b32
; EG: BFI_INT
define void @bfi_arg_arg_arg(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) nounwind {
  %bfi = call i32 @llvm.AMDGPU.bfi(i32 %src0, i32 %src1, i32 %src1) nounwind readnone
  store i32 %bfi, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfi_arg_arg_imm:
; SI: v_bfi_b32
; EG: BFI_INT
define void @bfi_arg_arg_imm(i32 addrspace(1)* %out, i32 %src0, i32 %src1) nounwind {
  %bfi = call i32 @llvm.AMDGPU.bfi(i32 %src0, i32 %src1, i32 123) nounwind readnone
  store i32 %bfi, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfi_arg_imm_arg:
; SI: v_bfi_b32
; EG: BFI_INT
define void @bfi_arg_imm_arg(i32 addrspace(1)* %out, i32 %src0, i32 %src2) nounwind {
  %bfi = call i32 @llvm.AMDGPU.bfi(i32 %src0, i32 123, i32 %src2) nounwind readnone
  store i32 %bfi, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfi_imm_arg_arg:
; SI: v_bfi_b32
; EG: BFI_INT
define void @bfi_imm_arg_arg(i32 addrspace(1)* %out, i32 %src1, i32 %src2) nounwind {
  %bfi = call i32 @llvm.AMDGPU.bfi(i32 123, i32 %src1, i32 %src2) nounwind readnone
  store i32 %bfi, i32 addrspace(1)* %out, align 4
  ret void
}

