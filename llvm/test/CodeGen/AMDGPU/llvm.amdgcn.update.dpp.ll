; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefix=VI -check-prefix=VI-OPT %s
; RUN: llc -O0 -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs -show-mc-encoding < %s | FileCheck -check-prefix=VI -check-prefix=VI-NOOPT %s

; VI-LABEL: {{^}}dpp_test:
; VI: v_mov_b32_e32 v0, s{{[0-9]+}}
; VI: v_mov_b32_e32 v1, s{{[0-9]+}}
; VI: s_nop 1
; VI: v_mov_b32_dpp v0, v1 quad_perm:[1,0,0,0] row_mask:0x1 bank_mask:0x1 bound_ctrl:0 ; encoding: [0xfa,0x02,0x00,0x7e,0x01,0x01,0x08,0x11]
define amdgpu_kernel void @dpp_test(i32 addrspace(1)* %out, i32 %in1, i32 %in2) {
  %tmp0 = call i32 @llvm.amdgcn.update.dpp.i32(i32 %in1, i32 %in2, i32 1, i32 1, i32 1, i1 1) #0
  store i32 %tmp0, i32 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32, i32, i32, i1) #0

attributes #0 = { nounwind readnone convergent }
