; RUN: llc -mtriple=amdgcn -mcpu=gfx1030 -filetype=obj < %s | llvm-objdump --triple=amdgcn--amdhsa --mcpu=gfx1030 -d - | FileCheck -check-prefix=OBJ %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1030 -show-mc-encoding < %s | FileCheck -check-prefix=ASM %s

declare i32 @llvm.amdgcn.permlane16(i32, i32, i32, i32, i1, i1)

; OBJ-LABEL: <permlane_op_sel>:
; OBJ: v_permlane16_b32 v0, v0, s7, s0 op_sel:[1,0]

; ASM-LABEL: permlane_op_sel:
; ASM: v_permlane16_b32 v0, v0, s7, s0 op_sel:[1,0] ; encoding: [0x00,0x08,0x77,0xd7,0x00,0x0f,0x00,0x00]
define amdgpu_kernel void @permlane_op_sel(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 1, i1 0)
  store i32 %v, ptr addrspace(1) %out
  ret void
}
