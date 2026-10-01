; RUN: llc -mtriple=amdgpu12.51 -filetype=obj < %s | llvm-objdump -d - | FileCheck --check-prefixes=CHECK,NORELAX %s
; RUN: llc -mtriple=amdgpu12.51 -filetype=obj --mc-relax-all < %s | llvm-objdump -d - | FileCheck --check-prefixes=CHECK,RELAXALL %s

; On subtargets with FeatureUseAddPC64Inst, branches are relaxed by the
; assembler instead of the BranchRelaxation pass. Out of range branches are
; relaxed into s_add_pc_i64 with a 32-bit literal, preceded by an inverted
; branch over it if the original branch was conditional.

; CHECK-LABEL: <short_forward_branch>:
; NORELAX:          s_cbranch_scc1 {{.*}} // [[#%.12X,BR:]]:
; NORELAX-NEXT:     s_nop 0 // [[#%.12X,BR+4]]:
; NORELAX-NOT:      s_add_pc_i64
; RELAXALL:         s_cbranch_scc0 2
; RELAXALL-NEXT:    s_add_pc_i64 lit(0x[[#%x,OFFSET:]]) // [[#%.12X,ADDPC:]]:
; RELAXALL:         s_load_b64 {{.*}} // [[#%.12X,ADDPC+8+OFFSET]]:
; CHECK:            s_endpgm
define amdgpu_kernel void @short_forward_branch(ptr addrspace(1) %arg, i32 %cnd) {
bb:
  %cmp = icmp eq i32 %cnd, 0
  br i1 %cmp, label %bb3, label %bb2

bb2:
  call void asm sideeffect "s_nop 0", ""()
  br label %bb3

bb3:
  store volatile i32 %cnd, ptr addrspace(1) %arg
  ret void
}

; CHECK-LABEL: <long_forward_branch>:
; CHECK:       s_cbranch_scc0 2
; CHECK-NEXT:  s_add_pc_i64 0x20000 // [[#%.12X,ADDPC:]]:
; CHECK:       s_load_b64 {{.*}} // [[#%.12X,ADDPC+8+0x20000]]:
; CHECK:       s_endpgm
define amdgpu_kernel void @long_forward_branch(ptr addrspace(1) %arg, i32 %cnd) {
bb:
  %cmp = icmp eq i32 %cnd, 0
  br i1 %cmp, label %bb3, label %bb2

bb2:
  call void asm sideeffect ".fill 32768, 4, 0xbf800000", ""()
  br label %bb3

bb3:
  store volatile i32 %cnd, ptr addrspace(1) %arg
  ret void
}

; The loop starts immediately after the s_mov_b32, so the s_add_pc_i64 must
; branch back to LOOP+4. Its literal is -0x2000c, sign-extended by the hardware,
; but the disassembler prints it zero-extended.
; CHECK-LABEL: <long_backward_branch>:
; CHECK:       s_mov_b32 vcc_lo, exec_lo // [[#%.12X,LOOP:]]:
; CHECK:       s_cbranch_vccz 2
; CHECK-NEXT:  s_add_pc_i64 0xfffdfff4 // [[#%.12X,LOOP+0x20008]]:
; CHECK-NEXT:  s_endpgm
define amdgpu_kernel void @long_backward_branch() {
entry:
  br label %loop

loop:
  call void asm sideeffect ".fill 32768, 4, 0xbf800000", ""()
  br label %loop
}
