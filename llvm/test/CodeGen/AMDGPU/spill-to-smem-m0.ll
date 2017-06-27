; RUN: llc -O0 -march=amdgcn -mcpu=fiji -amdgpu-spill-sgpr-to-smem=1 -verify-machineinstrs -stop-before=prologepilog < %s

; Spill to SMEM clobbers M0. Check that the implicit-def dead operand is present
; in the pseudo instructions.

; CHECK-LABEL: {{^}}spill_sgpr:
; CHECK: SI_SPILL_S32_SAVE {{.*}}, implicit-def dead %m0
; CHECK: SI_SPILL_S32_RESTORE {{.*}}, implicit-def dead %m0
define amdgpu_kernel void @spill_sgpr(i32 addrspace(1)* %out, i32 %in) #0 {
  %sgpr = call i32  asm sideeffect "; def $0", "=s" () #0
  %cmp = icmp eq i32 %in, 0
  br i1 %cmp, label %bb0, label %ret

bb0:
  call void asm sideeffect "; use $0", "s"(i32 %sgpr) #0
  br label %ret

ret:
  ret void
}

attributes #0 = { nounwind }
