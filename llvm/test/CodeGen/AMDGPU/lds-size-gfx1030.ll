; RUN: llc -mtriple=amdgpu10.30-mesa-mesa3d < %s | FileCheck %s --check-prefixes=CHECK,MESA
; RUN: llc -mtriple=amdgpu10.30-amd-amdpal < %s | FileCheck %s --check-prefixes=CHECK,PAL

; gfx1030 allocates LDS in 1024-byte blocks but encodes it in 512-byte units.
; 8252 bytes occupies 9216 bytes of LDS for occupancy, while its encoded size is
; 17 units (8704 bytes). Using the allocation granule for either encoding step
; would produce an incorrect block count or PAL metadata size.

@lds = addrspace(3) global [2063 x i32] poison, align 4

; CHECK-LABEL: lds_granularity:
; MESA: granulated_lds_size = 17
; CHECK: ; Occupancy: 7{{$}}
; PAL: .hardware_stages:
; PAL: .cs:
; PAL: .lds_size:       0x2200
define amdgpu_kernel void @lds_granularity(i32 %index, i32 %value) #0 {
  %ptr = getelementptr [2063 x i32], ptr addrspace(3) @lds, i32 0, i32 %index
  store volatile i32 %value, ptr addrspace(3) %ptr
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,64" }

!amdgpu.pal.metadata.msgpack = !{!0}
!0 = !{!"\81\AEamdpal.version\92\03\00"}
