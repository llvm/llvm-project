; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -o - %s | FileCheck -check-prefix=GFX1200 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -o - %s | FileCheck -check-prefix=GFX1250 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-icache-prefetch-size=16384 -o - %s | FileCheck -check-prefix=OVERRIDE-ENABLE %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -amdgpu-icache-prefetch-size=32768 -o - %s | FileCheck -check-prefix=OVERRIDE-DISABLE %s
; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -amdgpu-icache-prefetch-size=32769 -o /dev/null %s 2>&1 | FileCheck -check-prefix=INVALID-SIZE %s
; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -amdgpu-icache-prefetch-size=16385 -o /dev/null %s 2>&1 | FileCheck -check-prefix=INVALID-SIZE %s

; GFX12 defaults to a 32KiB I-cache and a 16KiB preferred INST_PREF_SIZE.
; MI450 (gfx1250) defaults to a 64KiB I-cache and a 32KiB preferred size.
; This 20KiB function is between those two preferred sizes.
define amdgpu_kernel void @size_between_defaults() {
; GFX1200-LABEL: size_between_defaults:
; GFX1200:       s_prefetch_inst_pc_rel
;
; GFX1250-LABEL: size_between_defaults:
; GFX1250-NOT:   s_prefetch_inst_pc_rel
; GFX1250:       s_endpgm
;
; OVERRIDE-ENABLE-LABEL: size_between_defaults:
; OVERRIDE-ENABLE:       s_prefetch_inst_pc_rel
;
; OVERRIDE-DISABLE-LABEL: size_between_defaults:
; OVERRIDE-DISABLE-NOT:   s_prefetch_inst_pc_rel
; OVERRIDE-DISABLE:       s_endpgm
  call void asm sideeffect ".space 20000", ""()
  ret void
}

; The GFX12 cache-size feature limits explicit prefetches to eight 4KiB slots.
define amdgpu_kernel void @gfx12_cache_size_limit() {
; GFX1200-LABEL: gfx12_cache_size_limit:
; GFX1200:       prefetchoffset(7,
; GFX1200-NOT:   prefetchoffset(8,
; GFX1200:       s_endpgm
  call void asm sideeffect ".space 65536", ""()
  ret void
}

; INVALID-SIZE: LLVM ERROR: -amdgpu-icache-prefetch-size must be a non-zero multiple of 128 bytes not exceeding 32768 bytes for gfx1200
