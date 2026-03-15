; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -O1 -global-isel < %s | FileCheck %s

; Test that V_INDIRECT_REG_READ_GPR_IDX expansion handles immediate index operands.
; The wave.reduce.umin with constant arguments folds to 0, which becomes an
; immediate index for the insertelement, triggering V_INDIRECT_REG_READ_GPR_IDX
; with an immediate operand.

; CHECK-LABEL: indirect_reg_read_imm_idx:
; CHECK: s_set_gpr_idx_on 0, gpr_idx(SRC0)
; CHECK-NEXT: v_mov_b32_e32
; CHECK-NEXT: s_set_gpr_idx_off
define amdgpu_kernel void @indirect_reg_read_imm_idx() {
entry:
  %vec = load <32 x i16>, ptr null, align 64
  %idx = call i32 @llvm.amdgcn.wave.reduce.umin.i32(i32 0, i32 0)
  %ins = insertelement <32 x i16> %vec, i16 0, i32 %idx
  store <32 x i16> %ins, ptr null, align 64
  ret void
}

declare i32 @llvm.amdgcn.wave.reduce.umin.i32(i32, i32 immarg)
