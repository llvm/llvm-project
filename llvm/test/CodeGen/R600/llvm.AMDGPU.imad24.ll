; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman -verify-machineinstrs < %s | FileCheck -check-prefix=CM -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s
; XUN: llc -march=r600 -mcpu=r600 -verify-machineinstrs < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s
; XUN: llc -march=r600 -mcpu=r770 -verify-machineinstrs < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s

; FIXME: Store of i32 seems to be broken pre-EG somehow?

declare i32 @llvm.AMDGPU.imad24(i32, i32, i32) nounwind readnone

; FUNC-LABEL: {{^}}test_imad24:
; SI: v_mad_i32_i24
; CM: MULADD_INT24
; R600: MULLO_INT
; R600: ADD_INT
define void @test_imad24(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) nounwind {
  %mad = call i32 @llvm.AMDGPU.imad24(i32 %src0, i32 %src1, i32 %src2) nounwind readnone
  store i32 %mad, i32 addrspace(1)* %out, align 4
  ret void
}

