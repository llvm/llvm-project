; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1010 < %s | FileCheck -check-prefix=GFX10-PAL %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1010 < %s | FileCheck -check-prefix=GFX10-MESA %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1100 < %s | FileCheck -check-prefix=GFX11-PAL %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1100 < %s | FileCheck -check-prefix=GFX11-MESA %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx950 < %s | FileCheck -check-prefix=GFX950-PAL %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx950 < %s | FileCheck -check-prefix=GFX950-MESA %s
; RUN: llc -mtriple=amdgcn-amd-amdpal -mcpu=gfx1210 < %s | FileCheck -check-prefix=GFX12-PAL %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1210 < %s | FileCheck -check-prefix=GFX12-MESA %s

; Check EXTRA_LDS_SIZE in SPI_SHADER_PGM_RSRC2_PS.

; GFX10-PAL: '0x2c0b (SPI_SHADER_PGM_RSRC2_PS)': 0x800

; GFX10-MESA: .long 45100
; GFX10-MESA-NEXT: .long 2048

; GFX11-PAL: '0x2c0b (SPI_SHADER_PGM_RSRC2_PS)': 0x400

; GFX11-MESA: .long 45100
; GFX11-MESA-NEXT: .long 1024

; GFX950-PAL: '0x2c0b (SPI_SHADER_PGM_RSRC2_PS)': 0x200

; GFX950-MESA: .long 45100
; GFX950-MESA-NEXT: .long 512

; GFX12-PAL: '0x2c0b (SPI_SHADER_PGM_RSRC2_PS)': 0x200

; GFX12-MESA: .long 45100
; GFX12-MESA-NEXT: .long 512

@lds = internal addrspace(3) global [4096 x i8] undef

define amdgpu_ps void @global_store_saddr_uniform_ptr_in_vgprs(i32 %voffset) {
  %ptr = getelementptr [4096 x i8], ptr addrspace(3) @lds, i32 0, i32 %voffset
  store i8 0, ptr addrspace(3) %ptr
  ret void
}
