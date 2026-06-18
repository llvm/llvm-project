; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -enable-ipra=0 < %s | FileCheck -check-prefix=CHECK %s

; A kernel that makes an indirect call is assigned the module-wide maximum
; register usage (any function is a potential callee). When the kernel's own
; SGPR budget is lowered (here via "amdgpu-num-sgpr") below the largest
; sibling function's explicit SGPR usage, the reported/emitted SGPR count must
; be clamped to what the kernel can actually allocate
; (ST.getMaxNumSGPRs() + extra SGPRs) instead of the inflated module maximum.
;
; For gfx942 the reserved SGPR count is 6 (FLAT_SCRATCH, XNACK, VCC), so with
; "amdgpu-num-sgpr"="40" the budget is getMaxNumSGPRs = min(40 - 6, 102) = 34
; explicit SGPRs, i.e. a total of 34 + 6 = 40 SGPRs. Without the cap the kernel
; would instead report the module maximum of 100 + 6 = 106.

; CHECK: .set .Luse_100_sgpr.numbered_sgpr, 100
define void @use_100_sgpr() #1 {
  call void asm sideeffect "", "~{s99}"() #0
  ret void
}

; CHECK: .set .Lindirect_call_low_sgpr_budget.has_indirect_call, 1
define amdgpu_kernel void @indirect_call_low_sgpr_budget(ptr %fptr) #0 {
  call void %fptr()
  ret void
}

; The module maximum still reflects the large sibling function.
; CHECK: .set amdgpu.max_num_sgpr, 100

; CHECK: amdhsa.kernels:
; CHECK-LABEL: .name: indirect_call_low_sgpr_budget
; CHECK: .sgpr_count: 40

attributes #0 = { nounwind noinline norecurse "amdgpu-num-sgpr"="40" }
attributes #1 = { nounwind noinline norecurse }
