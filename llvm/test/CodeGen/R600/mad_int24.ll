; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=EG --check-prefix=FUNC
; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck %s --check-prefix=CM --check-prefix=FUNC
; RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=SI --check-prefix=FUNC
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s --check-prefix=SI --check-prefix=FUNC

declare i32 @llvm.AMDGPU.imul24(i32, i32) nounwind readnone

; FUNC-LABEL: {{^}}i32_mad24:
; Signed 24-bit multiply is not supported on pre-Cayman GPUs.
; EG: MULLO_INT
; Make sure we aren't masking the inputs.
; CM-NOT: AND
; CM: MULADD_INT24
; SI-NOT: and
; SI: v_mad_i32_i24
define void @i32_mad24(i32 addrspace(1)* %out, i32 %a, i32 %b, i32 %c) {
entry:
  %0 = shl i32 %a, 8
  %a_24 = ashr i32 %0, 8
  %1 = shl i32 %b, 8
  %b_24 = ashr i32 %1, 8
  %2 = mul i32 %a_24, %b_24
  %3 = add i32 %2, %c
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @test_imul24
; SI: v_mad_i32_i24
define void @test_imul24(i32 addrspace(1)* %out, i32 %src0, i32 %src1, i32 %src2) nounwind {
  %mul = call i32 @llvm.AMDGPU.imul24(i32 %src0, i32 %src1) nounwind readnone
  %add = add i32 %mul, %src2
  store i32 %add, i32 addrspace(1)* %out, align 4
  ret void
}
