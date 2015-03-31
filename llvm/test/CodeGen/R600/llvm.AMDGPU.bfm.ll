; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i32 @llvm.AMDGPU.bfm(i32, i32) nounwind readnone

; FUNC-LABEL: {{^}}bfm_arg_arg:
; SI: v_bfm
; EG: BFM_INT
define void @bfm_arg_arg(i32 addrspace(1)* %out, i32 %src0, i32 %src1) nounwind {
  %bfm = call i32 @llvm.AMDGPU.bfm(i32 %src0, i32 %src1) nounwind readnone
  store i32 %bfm, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfm_arg_imm:
; SI: v_bfm
; EG: BFM_INT
define void @bfm_arg_imm(i32 addrspace(1)* %out, i32 %src0) nounwind {
  %bfm = call i32 @llvm.AMDGPU.bfm(i32 %src0, i32 123) nounwind readnone
  store i32 %bfm, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfm_imm_arg:
; SI: v_bfm
; EG: BFM_INT
define void @bfm_imm_arg(i32 addrspace(1)* %out, i32 %src1) nounwind {
  %bfm = call i32 @llvm.AMDGPU.bfm(i32 123, i32 %src1) nounwind readnone
  store i32 %bfm, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}bfm_imm_imm:
; SI: v_bfm
; EG: BFM_INT
define void @bfm_imm_imm(i32 addrspace(1)* %out) nounwind {
  %bfm = call i32 @llvm.AMDGPU.bfm(i32 123, i32 456) nounwind readnone
  store i32 %bfm, i32 addrspace(1)* %out, align 4
  ret void
}
