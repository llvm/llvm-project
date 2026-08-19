; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -filetype=null %s 2>&1 | FileCheck --implicit-check-not=warning -check-prefix=WARN %s

; Consumer test for the "amdgpu-accum-offset" register-file split
; cap applied in GCNSubtarget::getMaxNumVectorRegs.

;no amdgpu-agrp-alloc or amdgpu- accum-offset
; Budget 128, VGPR ceiling 64, AGPR ceiling 64

; no amdgpu-agpr-alloc attributes
; Budget 64, VGPR ceiling 48, AGPR Ceiling 16
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'split_96_32': desired occupancy was 8, final occupancy is 7
; WARN: warning: inline asm clobber list contains reserved registers: v48 at line 12
; WARN: warning: inline asm clobber list contains reserved registers: a16 at line 14
define amdgpu_kernel void @split_96_32() #0 {
  call void asm sideeffect "; c $0","~{v47}"(), !srcloc !{i32 11}
  call void asm sideeffect "; c $0","~{v48}"(), !srcloc !{i32 12}
  call void asm sideeffect "; c $0","~{a15}"(), !srcloc !{i32 13}
  call void asm sideeffect "; c $0","~{a16}"(), !srcloc !{i32 14}
  ret void
}
attributes #0 = {"amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-accum-offset"="48" }

; Budget 64, VGPR ceiling 32, AGPR ceiling 16.
; WARN: warning: inline asm clobber list contains reserved registers: v32 at line 22
; WARN: warning: inline asm clobber list contains reserved registers: a16 at line 24
define amdgpu_kernel void @split_32_16() #1 {
  call void asm sideeffect "; c $0","~{v31}"(), !srcloc !{i32 21}
  call void asm sideeffect "; c $0","~{v32}"(), !srcloc !{i32 22}
  call void asm sideeffect "; c $0","~{a15}"(), !srcloc !{i32 23}
  call void asm sideeffect "; c $0","~{a16}"(), !srcloc !{i32 24}
  ret void
}
attributes #1 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="16,16" "amdgpu-accum-offset"="32" }

; Budget 64, VGPR ceiling 24, AGPR ceiling 8.
; WARN: warning: inline asm clobber list contains reserved registers: v24 at line 32
; WARN: warning: inline asm clobber list contains reserved registers: a8 at line 34
define amdgpu_kernel void @split_24_8() #2 {
  call void asm sideeffect "; c $0","~{v23}"(), !srcloc !{i32 31}
  call void asm sideeffect "; c $0","~{v24}"(), !srcloc !{i32 32}
  call void asm sideeffect "; c $0","~{a7}"(), !srcloc !{i32 33}
  call void asm sideeffect "; c $0","~{a8}"(), !srcloc !{i32 34}
  ret void
}
attributes #2 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="8,8" "amdgpu-accum-offset"="24" }

; Budget 64, VGPR ceiling 16, AGPR ceiling 4
; WARN: warning: inline asm clobber list contains reserved registers: v16 at line 42
; WARN: warning: inline asm clobber list contains reserved registers: a4 at line 44
define amdgpu_kernel void @split_16_32() #3 {
  call void asm sideeffect "; c $0","~{v15}"(), !srcloc !{i32 41}
  call void asm sideeffect "; c $0","~{v16}"(), !srcloc !{i32 42}
  call void asm sideeffect "; c $0","~{a3}"(), !srcloc !{i32 43}
  call void asm sideeffect "; c $0","~{a4}"(), !srcloc !{i32 44}
  ret void
}
attributes #3 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-agpr-alloc"="4,4" "amdgpu-accum-offset"="16" }

; Budget 64, VGPR ceiling 24, AGPR ceiling 32
; WARN: warning: inline asm clobber list contains reserved registers: v24 at line 52
; WARN: warning: inline asm clobber list contains reserved registers: a32 at line 54
define amdgpu_kernel void @split_24_40() #4 {
  call void asm sideeffect "; c $0","~{v23}"(), !srcloc !{i32 51}
  call void asm sideeffect "; c $0","~{v24}"(), !srcloc !{i32 52}
  call void asm sideeffect "; c $0","~{a31}"(), !srcloc !{i32 53}
  call void asm sideeffect "; c $0","~{a32}"(), !srcloc !{i32 54}
  ret void
}
attributes #4 = { "amdgpu-waves-per-eu"="8,8" "amdgpu-flat-work-group-size"="64,64" "amdgpu-accum-offset"="24" }

; Budget 128, VGPR ceiling 64, AGPR ceiling 64.
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'pessimistic_split_64_64': desired occupancy was 4, final occupancy is 3
; WARN: warning: inline asm clobber list contains reserved registers: v64 at line 62
; WARN: warning: inline asm clobber list contains reserved registers: a64 at line 64
define amdgpu_kernel void @pessimistic_split_64_64() #5 {
  call void asm sideeffect "; c $0","~{v63}"(), !srcloc !{i32 61}
  call void asm sideeffect "; c $0","~{v64}"(), !srcloc !{i32 62}
  call void asm sideeffect "; c $0","~{a63}"(), !srcloc !{i32 63}
  call void asm sideeffect "; c $0","~{a64}"(), !srcloc !{i32 64}
  ret void
}
attributes #5 = {"amdgpu-waves-per-eu"="4,4" "amdgpu-flat-work-group-size"="256,256"}

; absent amdgpu-accum-offset attribute
; Budget 128, VGPR ceiling 64, AGPR ceiling 16
; WARN: warning: inline asm clobber list contains reserved registers: v64 at line 72
; WARN: warning: inline asm clobber list contains reserved registers: a16 at line 74
define amdgpu_kernel void @absent_accum_offset_64_16() #6 {
  call void asm sideeffect "; c $0","~{v63}"(), !srcloc !{i32 71}
  call void asm sideeffect "; c $0","~{v64}"(), !srcloc !{i32 72}
  call void asm sideeffect "; c $0","~{a15}"(), !srcloc !{i32 73}
  call void asm sideeffect "; c $0","~{a16}"(), !srcloc !{i32 74}
  ret void
}

attributes #6 = {"amdgpu-waves-per-eu"="4,4" "amdgpu-flat-work-group-size"="256,256" "amdgpu-agpr-alloc"="16,16" }

; Budget 128, VGPR ceiling 64, AGPR ceiling 32
; WARN: warning: inline asm clobber list contains reserved registers: v64 at line 82
; WARN: warning: inline asm clobber list contains reserved registers: a32 at line 84
define amdgpu_kernel void @agpr_alloc_16_16_64_32() #7 {
  call void asm sideeffect "; c $0","~{v63}"(), !srcloc !{i32 81}
  call void asm sideeffect "; c $0","~{v64}"(), !srcloc !{i32 82}
  call void asm sideeffect "; c $0","~{a31}"(), !srcloc !{i32 83}
  call void asm sideeffect "; c $0","~{a32}"(), !srcloc !{i32 84}
  ret void
}
attributes #7 = {"amdgpu-waves-per-eu"="4,4" "amdgpu-flat-work-group-size"="256,256" "amdgpu-agpr-alloc"="16,32" }

; Budget 128, VGPR ceiling 56, AGPR ceiling 72
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'agpr_alloc_70_58_70': desired occupancy was 4, final occupancy is 3
; WARN: warning: inline asm clobber list contains reserved registers: v56 at line 92
; WARN: warning: inline asm clobber list contains reserved registers: a72 at line 94
define amdgpu_kernel void @agpr_alloc_70_58_70() #8 {
  call void asm sideeffect "; c $0","~{v55}"(), !srcloc !{i32 91}
  call void asm sideeffect "; c $0","~{v56}"(), !srcloc !{i32 92}
  call void asm sideeffect "; c $0","~{a61}"(), !srcloc !{i32 93}
  call void asm sideeffect "; c $0","~{a72}"(), !srcloc !{i32 94}
  ret void
}
attributes #8 = {"amdgpu-waves-per-eu"="4,4" "amdgpu-flat-work-group-size"="256,256" "amdgpu-agpr-alloc"="72" }

;more restraining attribute takes priority over amdgpu-accum-offset attribute
; Budget 128, VGPR ceiling 80, AGPR Ceiling 48
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'agpr_alloc_48_80_48': desired occupancy was 4, final occupancy is 3
; WARN: warning: inline asm clobber list contains reserved registers: v80 at line 102
; WARN: warning: inline asm clobber list contains reserved registers: a48 at line 104
define amdgpu_kernel void @agpr_alloc_48_80_48() #9 {
  call void asm sideeffect "; c $0","~{v79}"(), !srcloc !{i32 101}
  call void asm sideeffect "; c $0","~{v80}"(), !srcloc !{i32 102}
  call void asm sideeffect "; c $0","~{a47}"(), !srcloc !{i32 103}
  call void asm sideeffect "; c $0","~{a48}"(), !srcloc !{i32 104}
  ret void
}
attributes #9 = {"amdgpu-waves-per-eu"="4,4" "amdgpu-flat-work-group-size"="256,256" "amdgpu-accum-offset"="100" "amdgpu-agpr-alloc"="48" }

;lower amdgpu-accum-offset attribute could allow agpr more space
; Budget 128, VGPR ceiling 100, AGPR ceiling 26
; WARN: warning: <unknown>:0:0: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in 'agpr_alloc_10_26_100_26': desired occupancy was 4, final occupancy is 3
; WARN: warning: inline asm clobber list contains reserved registers: v100 at line 112
; WARN: warning: inline asm clobber list contains reserved registers: a26 at line 114
define amdgpu_kernel void @agpr_alloc_10_26_100_26() #10 {
  call void asm sideeffect "; c $0","~{v99}"(), !srcloc !{i32 111}
  call void asm sideeffect "; c $0","~{v100}"(), !srcloc !{i32 112}
  call void asm sideeffect "; c $0","~{a25}"(), !srcloc !{i32 113}
  call void asm sideeffect "; c $0","~{a26}"(), !srcloc !{i32 114}
  ret void
}
attributes #10 = {"amdgpu-waves-per-eu"="4,4" "amdgpu-flat-work-group-size"="256,256" "amdgpu-accum-offset"="100" "amdgpu-agpr-alloc"="10,26" }