; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman -verify-machineinstrs < %s | FileCheck -check-prefix=CM -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s

declare i32 @llvm.AMDGPU.imul24(i32, i32) nounwind readnone

; FUNC-LABEL: {{^}}test_imul24:
; SI: v_mul_i32_i24
; CM: MUL_INT24
; R600: MULLO_INT
define void @test_imul24(i32 addrspace(1)* %out, i32 %src0, i32 %src1) nounwind {
  %mul = call i32 @llvm.AMDGPU.imul24(i32 %src0, i32 %src1) nounwind readnone
  store i32 %mul, i32 addrspace(1)* %out, align 4
  ret void
}
