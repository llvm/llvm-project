; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -O2 -pass-remarks-missed=si-pin-vgpr < %s 2>&1 | FileCheck %s
; RUN: llc -global-isel -global-isel-abort=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -O2 -pass-remarks-missed=si-pin-vgpr < %s 2>&1 | FileCheck %s

; When the pinned values' own peak demand exceeds the VGPR budget, the pins
; cannot be honored. SIPinVGPR detects this up front, emits an optimization
; remark (only under -pass-remarks-missed; never an unconditional user warning,
; since the intrinsic is compiler-internal), and leaves the intervals spillable
; so allocation degrades gracefully (spill to scratch) instead of the allocator
; aborting with "ran out of registers".

; Three pinned <8 x i32> values are all live across the stores: 24 VGPRs of
; pinned demand against an "amdgpu-num-vgpr"="20" budget. gfx1100 has no AGPRs,
; so the VGPR-only budget equals the requested 20 (no AGPR split / rounding).

; CHECK: remark: {{.*}}VGPR pinning could not be honored: pinned values need 24 VGPRs but only 20 are available; affected values will be spilled

; The function must still compile to valid (spilled) code rather than failing.
; CHECK-LABEL: overflow:
; CHECK: scratch_store
; CHECK: s_setpc_b64

define void @overflow(ptr addrspace(1) %p) #0 {
  %a = call <8 x i32> asm sideeffect "; def a", "=v"()
  call void @llvm.amdgcn.internal.vgpr.pin.v8i32(<8 x i32> %a)
  %b = call <8 x i32> asm sideeffect "; def b", "=v"()
  call void @llvm.amdgcn.internal.vgpr.pin.v8i32(<8 x i32> %b)
  %c = call <8 x i32> asm sideeffect "; def c", "=v"()
  call void @llvm.amdgcn.internal.vgpr.pin.v8i32(<8 x i32> %c)
  store volatile <8 x i32> %a, ptr addrspace(1) %p
  store volatile <8 x i32> %b, ptr addrspace(1) %p
  store volatile <8 x i32> %c, ptr addrspace(1) %p
  ret void
}

; A pinned set that fits the budget must compile with no remark.
; CHECK-NOT: remark:
; CHECK-LABEL: fits:

define void @fits(ptr addrspace(1) %p) #1 {
  %a = call <8 x i32> asm sideeffect "; def a", "=v"()
  call void @llvm.amdgcn.internal.vgpr.pin.v8i32(<8 x i32> %a)
  %b = call <8 x i32> asm sideeffect "; def b", "=v"()
  call void @llvm.amdgcn.internal.vgpr.pin.v8i32(<8 x i32> %b)
  %c = call <8 x i32> asm sideeffect "; def c", "=v"()
  call void @llvm.amdgcn.internal.vgpr.pin.v8i32(<8 x i32> %c)
  store volatile <8 x i32> %a, ptr addrspace(1) %p
  store volatile <8 x i32> %b, ptr addrspace(1) %p
  store volatile <8 x i32> %c, ptr addrspace(1) %p
  ret void
}

declare void @llvm.amdgcn.internal.vgpr.pin.v8i32(<8 x i32>)

attributes #0 = { nounwind "amdgpu-num-vgpr"="20" }
attributes #1 = { nounwind "amdgpu-num-vgpr"="32" }
