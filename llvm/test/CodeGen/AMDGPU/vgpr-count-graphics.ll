; RUN: llc -mcpu=gfx1200 < %s | FileCheck %s
target triple = "amdgcn--amdpal"

@global = addrspace(1) global i32 poison, align 4

; CHECK-LABEL: amdpal.pipelines:

; Neither uses not writes a VGPR, but the hardware initializes the VGPRs that the kernel receives, so they count as used.
; CHECK-LABEL: .entry_point_symbol: kernel_use
; CHECK: .vgpr_count:     0x20
define amdgpu_cs void @kernel_use([32 x i32] %args) {
entry:
  %a = extractvalue [32 x i32] %args, 14
  store i32 %a, ptr addrspace(1) @global
  ret void
}

; Neither uses not writes a VGPR
; CHECK-LABEL: gfx_func:
; CHECK: .vgpr_count:     0x20
define amdgpu_gfx [32 x i32] @gfx_func([32 x i32] %args) {
entry:
  ret [32 x i32] %args
}
