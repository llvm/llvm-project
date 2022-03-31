// RUN: llvm-mc -arch=amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck -check-prefixes=GFX90A %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx908 %s 2>&1 | FileCheck -check-prefixes=GFX908 %s
// Based on sym_kernel_scope.s

.byte .kernel.agpr_count
// CHECK: .byte 0
.byte .kernel.vgpr_count
// CHECK: .byte 0

    v_accvgpr_write_b32 a0, v6
    v_accvgpr_read_b32 v3, a3
    s_endpgm
.byte .kernel.agpr_count
// GFX90A: .byte 4
// GFX908: .byte 4
.byte .kernel.vgpr_count
// GFX90A: .byte 12
// GFX908: .byte 7

.amdgpu_hsa_kernel K1
K1:
.byte .kernel.agpr_count
// CHECK: .byte 0
.byte .kernel.vgpr_count
// CHECK: .byte 0
    v_accvgpr_write_b32 a44, v6
    s_endpgm
.byte .kernel.agpr_count
// GFX90A: .byte 45
// GFX908: .byte 45
.byte .kernel.vgpr_count
// GFX90A: .byte 53
// GFX908: .byte 45

.amdgpu_hsa_kernel K2
.byte .kernel.agpr_count
// CHECK: .byte 0
.byte .kernel.vgpr_count
// CHECK: .byte 0
K2:
    v_mfma_f32_4x4x1f32 a[0:3], v1, v0, a[0:3] cbsz:1 abid:2 blgp:3
    s_endpgm
.byte .kernel.agpr_count
// GFX90A: .byte 4
// GFX908: .byte 4
.byte .kernel.vgpr_count
// GFX90A: .byte 8
// GFX908: .byte 4

.text
.amdgpu_hsa_kernel K3
K3:
    v_accvgpr_read_b32 v[0], a0
    v_mfma_f32_16x16x1f32 a[0:15], v1, v0, a[0:15] cbsz:1 abid:2 blgp:3
    s_endpgm

.byte .kernel.agpr_count
// GFX90A: .byte 16
// GFX908: .byte 16
.byte .kernel.vgpr_count
// GFX90A: .byte 20
// GFX908: .byte 16
