// RUN: llvm-mc -arch=amdgcn %s 2>&1 | FileCheck %s

.byte .kernel.sgpr_count
// CHECK: .byte 0
.byte .kernel.vgpr_count
// CHECK: .byte 0
    v_mov_b32_e32 v5, s8
    s_endpgm
.byte .kernel.sgpr_count
// CHECK: .byte 9
.byte .kernel.vgpr_count
// CHECK: .byte 6

.amdgpu_hsa_kernel K1
K1:
.byte .kernel.sgpr_count
// CHECK: .byte 0
.byte .kernel.vgpr_count
// CHECK: .byte 0
    v_mov_b32_e32 v1, s86
    s_endpgm
.byte .kernel.sgpr_count
// CHECK: .byte 87
.byte .kernel.vgpr_count
// CHECK: .byte 2

.amdgpu_hsa_kernel K2
.byte .kernel.sgpr_count
// CHECK: .byte 0
.byte .kernel.vgpr_count
// CHECK: .byte 0
K2:
    s_load_dwordx8 s[16:23], s[0:1], 0x0
    v_mov_b32_e32 v0, v0
    s_endpgm
.byte .kernel.sgpr_count
// CHECK: .byte 24
.byte .kernel.vgpr_count
// CHECK: .byte 1

.text
.amdgpu_hsa_kernel K3
K3:
A = .kernel.vgpr_count
    v_mov_b32_e32 v[A], s0
B = .kernel.vgpr_count
    v_mov_b32_e32 v[B], s0
    v_mov_b32_e32 v[B], v[A]
C = .kernel.vgpr_count
    v_mov_b32_e32 v[C], v[A]
D = .kernel.sgpr_count + 3 // align
E = D + 4
    s_load_dwordx4 s[D:D+3], s[E:E+1], 0x0
    s_endpgm

.byte .kernel.sgpr_count
// CHECK: .byte 10
.byte .kernel.vgpr_count
// CHECK: .byte 3
