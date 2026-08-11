; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -filetype=null %s 2>&1 | FileCheck --implicit-check-not=warning -check-prefix=WARN %s

; Consumer-isolation test for the "amdgpu-register-budget" register-file split
; cap applied in GCNSubtarget::getMaxNumVectorRegs. The attribute is written by
; hand here so the cap is exercised directly, independent of the
; amdgpu-attributor pass that normally produces it.
;
; The attribute is a pair "VGPRs,AGPRs" of absolute ceilings: VGPRs caps the
; architected VGPRs, AGPRs caps the accumulation registers, and the two axes are
; applied independently.
; The consumer only reads amdgpu-register-budget when amdgpu-agpr-alloc is
; present and non-default, so every kernel below sets both.
;
; All kernels use waves-per-eu=8,8 and flat-work-group-size=64,64, giving a
; total vector-register budget of 64 on gfx90a. A register above the resulting
; cap becomes reserved, so clobbering it in inline asm emits a warning keyed by
; !srcloc; a register just below the cap must NOT warn (enforced by
; --implicit-check-not=warning).

; Budget 64, VGPR ceiling 32, AGPR ceiling 16.
; WARN: warning: inline asm clobber list contains reserved registers: v32 at line 12
; WARN: warning: inline asm clobber list contains reserved registers: a16 at line 14
define amdgpu_kernel void @split_32_16() #0 {
  call void asm sideeffect "; c $0","~{v31}"(), !srcloc !{i32 11}
  call void asm sideeffect "; c $0","~{v32}"(), !srcloc !{i32 12}
  call void asm sideeffect "; c $0","~{a15}"(), !srcloc !{i32 13}
  call void asm sideeffect "; c $0","~{a16}"(), !srcloc !{i32 14}
  ret void
}
attributes #0 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="16" "amdgpu-register-budget"="32,16" }

; Budget 64, VGPR ceiling 24, AGPR ceiling 8.
; WARN: warning: inline asm clobber list contains reserved registers: v24 at line 22
; WARN: warning: inline asm clobber list contains reserved registers: a8 at line 24
define amdgpu_kernel void @split_24_8() #1 {
  call void asm sideeffect "; c $0","~{v23}"(), !srcloc !{i32 21}
  call void asm sideeffect "; c $0","~{v24}"(), !srcloc !{i32 22}
  call void asm sideeffect "; c $0","~{a7}"(), !srcloc !{i32 23}
  call void asm sideeffect "; c $0","~{a8}"(), !srcloc !{i32 24}
  ret void
}
attributes #1 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="8" "amdgpu-register-budget"="24,8" }

; VGPR ceiling and AGPR ceiling come from opposite ends: a tight VGPR ceiling of
; 16 with a generous AGPR ceiling of 32. Confirms the two axes are capped
; independently.
; WARN: warning: inline asm clobber list contains reserved registers: v16 at line 32
; WARN: warning: inline asm clobber list contains reserved registers: a32 at line 34
define amdgpu_kernel void @split_16_32() #2 {
  call void asm sideeffect "; c $0","~{v15}"(), !srcloc !{i32 31}
  call void asm sideeffect "; c $0","~{v16}"(), !srcloc !{i32 32}
  call void asm sideeffect "; c $0","~{a31}"(), !srcloc !{i32 33}
  call void asm sideeffect "; c $0","~{a32}"(), !srcloc !{i32 34}
  ret void
}
attributes #2 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="4" "amdgpu-register-budget"="16,32" }
