; Check the gfx1200 default initial prefetch size.
; RUN: llc -mtriple=amdgpu12.00-amd-amdhsa -o - %s | FileCheck -check-prefix=GFX1200 %s
; Check overriding only the explicit-prefetch threshold.
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -amdgpu-icache-prefetch-threshold=16384 -o - %s | FileCheck -check-prefix=THRESHOLD-16K %s
; Check independently overriding the initial prefetch size.
; RUN: llc -mtriple=amdgpu12.50-amd-amdhsa -amdgpu-icache-prefetch-initial-size=16384 -amdgpu-icache-prefetch-threshold=16384 -o - %s | FileCheck -check-prefix=EQUAL-16K %s

; The default threshold is the 32640-byte descriptor capacity.
define amdgpu_kernel void @above_default_threshold() {
; GFX1200-LABEL: above_default_threshold:
; GFX1200:       s_prefetch_inst_pc_rel prefetchoffset(128,
; GFX1200:       .amdhsa_inst_pref_size 128
  call void asm sideeffect ".space 40000", ""()
  ret void
}

; Lowering only the threshold activates explicit prefetching while retaining
; the default 8KiB initial descriptor coverage. Setting initial == threshold
; is also valid and changes the descriptor coverage independently.
define amdgpu_kernel void @between_override_thresholds() {
; THRESHOLD-16K-LABEL: between_override_thresholds:
; THRESHOLD-16K:       s_prefetch_inst_pc_rel prefetchoffset(64,
; THRESHOLD-16K:       .amdhsa_inst_pref_size 64
;
; EQUAL-16K-LABEL: between_override_thresholds:
; EQUAL-16K:       s_prefetch_inst_pc_rel prefetchoffset(128,
; EQUAL-16K:       .amdhsa_inst_pref_size 128
  call void asm sideeffect ".space 20000", ""()
  ret void
}
