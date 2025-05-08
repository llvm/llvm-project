; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1300 <%s | FileCheck %s --check-prefixes=CHECK

; CHECK: .gs:
; CHECK: .entry_point_symbol: gs_shader_granularity_256DW
; CHECK-NEXT: .lds_size:       0x5c00


@LDS.GS = external addrspace(3) global [u0x5808 x i8], align 4

define dllexport amdgpu_gs void @gs_shader_granularity_256DW() {
  %ptr = getelementptr i8, ptr addrspace(3) @LDS.GS, i8 0
  store i8 0, ptr addrspace(3) %ptr, align 4
  ret void
}

!amdgpu.pal.metadata.msgpack = !{!0}

!0 = !{!"\81\AEamdpal.version\92\03\00"}
