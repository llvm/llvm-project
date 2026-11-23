; RUN: llc -mtriple=amdgpu9.42-mesa-mesa3d < %s | FileCheck %s --check-prefix=GFX942
; RUN: llc -mtriple=amdgpu9.50-mesa-mesa3d < %s | FileCheck %s --check-prefix=GFX950
; RUN: llc -mtriple=amdgpu9.4-mesa-mesa3d < %s | FileCheck %s --check-prefix=GENERIC

@lds1280 = addrspace(3) global [320 x i32] poison, align 4
@lds1284 = addrspace(3) global [321 x i32] poison, align 4
@lds65536 = addrspace(3) global [16384 x i32] poison, align 4

; GFX942-LABEL: one_granule:
; GFX942: granulated_lds_size = 3
; GFX950-LABEL: one_granule:
; GFX950: granulated_lds_size = 1
; GENERIC-LABEL: one_granule:
; GENERIC: granulated_lds_size = 1
define amdgpu_kernel void @one_granule(i32 %index, i32 %value) {
  %ptr = getelementptr [320 x i32], ptr addrspace(3) @lds1280, i32 0, i32 %index
  store volatile i32 %value, ptr addrspace(3) %ptr
  ret void
}

; GFX942-LABEL: two_granules:
; GFX942: granulated_lds_size = 3
; GFX950-LABEL: two_granules:
; GFX950: granulated_lds_size = 2
; GENERIC-LABEL: two_granules:
; GENERIC: granulated_lds_size = 2
define amdgpu_kernel void @two_granules(i32 %index, i32 %value) {
  %ptr = getelementptr [321 x i32], ptr addrspace(3) @lds1284, i32 0, i32 %index
  store volatile i32 %value, ptr addrspace(3) %ptr
  ret void
}

; A 64 KiB work-group must fit on the generic target.
; GFX942-LABEL: full_lds:
; GFX942: ; Occupancy: 4{{$}}
; GFX950-LABEL: full_lds:
; GFX950: ; Occupancy: 8{{$}}
; GENERIC-LABEL: full_lds:
; GENERIC: ; Occupancy: 4{{$}}
define amdgpu_kernel void @full_lds(i32 %index, i32 %value) #0 {
  %ptr = getelementptr [16384 x i32], ptr addrspace(3) @lds65536, i32 0, i32 %index
  store volatile i32 %value, ptr addrspace(3) %ptr
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1024,1024" }
