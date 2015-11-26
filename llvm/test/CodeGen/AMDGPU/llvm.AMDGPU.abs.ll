; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i32 @llvm.AMDGPU.abs(i32) nounwind readnone

; Legacy name
declare i32 @llvm.AMDIL.abs.i32(i32) nounwind readnone

; FUNC-LABEL: {{^}}s_abs_i32:
; SI: s_abs_i32

; EG: SUB_INT
; EG: MAX_INT
define void @s_abs_i32(i32 addrspace(1)* %out, i32 %src) nounwind {
  %abs = call i32 @llvm.AMDGPU.abs(i32 %src) nounwind readnone
  store i32 %abs, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_abs_i32:
; SI: v_sub_i32_e32
; SI: v_max_i32_e32
; SI: s_endpgm

; EG: SUB_INT
; EG: MAX_INT
define void @v_abs_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %src) nounwind {
  %val = load i32, i32 addrspace(1)* %src, align 4
  %abs = call i32 @llvm.AMDGPU.abs(i32 %val) nounwind readnone
  store i32 %abs, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}abs_i32_legacy_amdil:
; SI: v_sub_i32_e32
; SI: v_max_i32_e32
; SI: s_endpgm

; EG: SUB_INT
; EG: MAX_INT
define void @abs_i32_legacy_amdil(i32 addrspace(1)* %out, i32 addrspace(1)* %src) nounwind {
  %val = load i32, i32 addrspace(1)* %src, align 4
  %abs = call i32 @llvm.AMDIL.abs.i32(i32 %val) nounwind readnone
  store i32 %abs, i32 addrspace(1)* %out, align 4
  ret void
}
