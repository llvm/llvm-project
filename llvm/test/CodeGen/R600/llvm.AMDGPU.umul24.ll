; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=r600 -mcpu=cayman -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; XUN: llc -march=r600 -mcpu=r600 -verify-machineinstrs < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s
; XUN: llc -march=r600 -mcpu=r770 -verify-machineinstrs < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s

declare i32 @llvm.AMDGPU.umul24(i32, i32) nounwind readnone

; FUNC-LABEL: {{^}}test_umul24:
; SI: v_mul_u32_u24
; R600: MUL_UINT24
; R600: MULLO_UINT
define void @test_umul24(i32 addrspace(1)* %out, i32 %src0, i32 %src1) nounwind {
  %mul = call i32 @llvm.AMDGPU.umul24(i32 %src0, i32 %src1) nounwind readnone
  store i32 %mul, i32 addrspace(1)* %out, align 4
  ret void
}
