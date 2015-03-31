; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare float @llvm.AMDGPU.lrp(float, float, float) nounwind readnone

; FUNC-LABEL: {{^}}test_lrp:
; SI: v_sub_f32
; SI: v_mad_f32
define void @test_lrp(float addrspace(1)* %out, float %src0, float %src1, float %src2) nounwind {
  %mad = call float @llvm.AMDGPU.lrp(float %src0, float %src1, float %src2) nounwind readnone
  store float %mad, float addrspace(1)* %out, align 4
  ret void
}
