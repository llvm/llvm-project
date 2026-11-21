; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -verify-machineinstrs < %s | FileCheck %s

; v_permlane32_swap has two tied operands. Here both read lanes of the same
; 128-bit load, so two-address inserts two copies of %v's subregisters. Each
; subrange of %v must end at the copy that reads its lanes; ending all of them
; at the last copy left a live segment ending at an instruction that does not
; read it, which the machine verifier rejects.

; CHECK-LABEL: tied_subreg_lanes:
; CHECK: global_load_dwordx4
; CHECK: v_permlane32_swap_b32
define amdgpu_kernel void @tied_subreg_lanes(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gp = getelementptr <4 x i32>, ptr addrspace(1) %p, i32 %tid
  %v = load <4 x i32>, ptr addrspace(1) %gp
  %e0 = extractelement <4 x i32> %v, i64 0
  %e2 = extractelement <4 x i32> %v, i64 2
  %s = call { i32, i32 } @llvm.amdgcn.permlane32.swap(i32 %e0, i32 %e2, i1 false, i1 false)
  %r0 = extractvalue { i32, i32 } %s, 0
  %r1 = extractvalue { i32, i32 } %s, 1
  %x = add i32 %r0, %r1
  %gq = getelementptr i32, ptr addrspace(1) %q, i32 %tid
  store i32 %x, ptr addrspace(1) %gq
  ret void
}
