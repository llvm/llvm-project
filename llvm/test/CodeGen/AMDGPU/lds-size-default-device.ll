; RUN: llc -mtriple=amdgcn-mesa-mesa3d < %s | FileCheck %s --check-prefixes=CHECK,MESA
; RUN: llc -mtriple=amdgcn-amd-amdpal < %s | FileCheck %s --check-prefixes=CHECK,PAL
; RUN: llc -mtriple=amdgcn-amd-amdhsa < %s | FileCheck %s --check-prefixes=CHECK,HSA

; The legacy generic and generic-hsa targets have no LDS encoding granularity
; feature. Code generation still uses its default 256-byte encoding granule,
; rounding 260 bytes up to two units (512 bytes) for registers and PAL metadata.
; HSA metadata records the unrounded size in bytes.

@lds = addrspace(3) global [65 x i32] poison, align 4

; CHECK-LABEL: lds_granularity:
; MESA: granulated_lds_size = 2
; HSA: .amdhsa_group_segment_fixed_size 260
; PAL: .hardware_stages:
; PAL: .cs:
; PAL: .lds_size:       0x200
define amdgpu_kernel void @lds_granularity(i32 %index, i32 %value) {
  %ptr = getelementptr [65 x i32], ptr addrspace(3) @lds, i32 0, i32 %index
  store volatile i32 %value, ptr addrspace(3) %ptr
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}

!amdgpu.pal.metadata.msgpack = !{!1}
!1 = !{!"\81\AEamdpal.version\92\03\00"}
