; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -o - %s | FileCheck -check-prefix=GFX1200 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -o - %s | FileCheck -check-prefix=GFX1250 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-icache-prefetch-initial-size=16384 -o - %s | FileCheck -check-prefix=INITIAL-16K %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-icache-prefetch-initial-size=32640 -o - %s | FileCheck -check-prefix=MAX-INITIAL %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-icache-prefetch-threshold=49152 -o - %s | FileCheck -check-prefix=THRESHOLD-48K %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-icache-prefetch-threshold=16384 -o - %s | FileCheck -check-prefix=THRESHOLD-16K %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-icache-prefetch-initial-size=16384 -amdgpu-icache-prefetch-threshold=16384 -o - %s | FileCheck -check-prefix=EQUAL-16K %s
; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-icache-prefetch-initial-size=0 -o /dev/null %s 2>&1 | FileCheck -check-prefix=INVALID-INITIAL %s
; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-icache-prefetch-initial-size=8193 -o /dev/null %s 2>&1 | FileCheck -check-prefix=INVALID-INITIAL %s
; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-icache-prefetch-initial-size=32768 -o /dev/null %s 2>&1 | FileCheck -check-prefix=INVALID-INITIAL %s
; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-icache-prefetch-threshold=0 -o /dev/null %s 2>&1 | FileCheck -check-prefix=INVALID-THRESHOLD %s
; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-icache-prefetch-threshold=32769 -o /dev/null %s 2>&1 | FileCheck -check-prefix=INVALID-THRESHOLD %s
; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-icache-prefetch-threshold=65664 -o /dev/null %s 2>&1 | FileCheck -check-prefix=INVALID-THRESHOLD %s
; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-icache-prefetch-initial-size=16384 -amdgpu-icache-prefetch-threshold=8192 -o /dev/null %s 2>&1 | FileCheck -check-prefix=INVALID-ORDER %s

; The default threshold is the 32640-byte descriptor capacity on both targets.
; gfx1200 initially prefetches 16KiB, while gfx1250 initially prefetches 8KiB.
define amdgpu_kernel void @above_default_threshold() {
; GFX1200-LABEL: above_default_threshold:
; GFX1200:       s_prefetch_inst_pc_rel prefetchoffset(128,
; GFX1200:       .amdhsa_inst_pref_size 128
;
; GFX1250-LABEL: above_default_threshold:
; GFX1250:       s_prefetch_inst_pc_rel prefetchoffset(64,
; GFX1250:       .amdhsa_inst_pref_size 64
;
; INITIAL-16K-LABEL: above_default_threshold:
; INITIAL-16K:       s_prefetch_inst_pc_rel prefetchoffset(128,
; INITIAL-16K:       .amdhsa_inst_pref_size 128
;
; MAX-INITIAL-LABEL: above_default_threshold:
; MAX-INITIAL:       s_prefetch_inst_pc_rel prefetchoffset(255,
; MAX-INITIAL:       .amdhsa_inst_pref_size 255
;
; THRESHOLD-48K-LABEL: above_default_threshold:
; THRESHOLD-48K-NOT:   s_prefetch_inst_pc_rel
; THRESHOLD-48K:       s_endpgm
; THRESHOLD-48K:       .amdhsa_inst_pref_size ((instprefsize(
  call void asm sideeffect ".space 40000", ""()
  ret void
}

; The gfx1250 prologue and s_endpgm add 32 bytes, making these functions
; exactly 32640 bytes and four bytes over 32640 bytes respectively.
define amdgpu_kernel void @at_default_threshold() {
; GFX1250-LABEL: at_default_threshold:
; GFX1250-NOT:   s_prefetch_inst_pc_rel
; GFX1250:       s_endpgm
  call void asm sideeffect ".space 32608", ""()
  ret void
}

define amdgpu_kernel void @above_default_threshold_by_four() {
; GFX1250-LABEL: above_default_threshold_by_four:
; GFX1250:       s_prefetch_inst_pc_rel prefetchoffset(64,
  call void asm sideeffect ".space 32612", ""()
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

; INITIAL-16K-LABEL: cache_size_limit:
; INITIAL-16K:          s_mov_b64
; INITIAL-16K-NEXT:     v_nop
; INITIAL-16K-NEXT:     global_prefetch_b8
; INITIAL-16K-COUNT-12: s_prefetch_inst_pc_rel
; INITIAL-16K-NEXT:     ;;#ASMSTART
; GFX1250-LABEL:     cache_size_limit:
; GFX1250:           s_mov_b64
; GFX1250-NEXT:      v_nop
; GFX1250-NEXT:      global_prefetch_b8
; GFX1250-COUNT-14:  s_prefetch_inst_pc_rel
; GFX1250-NEXT:      ;;#ASMSTART
define amdgpu_kernel void @cache_size_limit() {
  call void asm sideeffect ".space 65536", ""()
  ret void
}

; INVALID-INITIAL: LLVM ERROR: -amdgpu-icache-prefetch-initial-size must be a non-zero multiple of 128 bytes not exceeding 32640 bytes for gfx1250
; INVALID-THRESHOLD: LLVM ERROR: -amdgpu-icache-prefetch-threshold must be a non-zero multiple of 128 bytes not exceeding 65536 bytes for gfx1250
; INVALID-ORDER: LLVM ERROR: -amdgpu-icache-prefetch-initial-size must not exceed -amdgpu-icache-prefetch-threshold for gfx1250
