// RUN: llvm-mc -triple=amdgcn -mcpu=gfx1300 -show-encoding %s | FileCheck --check-prefix=GFX13 %s

v_send_vgpr_next_b32 v1, v2, v3
// GFX13: v_send_vgpr_next_b32 v1, v2, v3 sema_wave_id:0 sema_wave_id_refl:0 wait_va_vdst:0 ; encoding: [0x01,0x00,0x00,0xfa,0x03,0x04,0x20,0x00]

v_send_vgpr_next_b32 v1023, v1023, v1023
// GFX13: v_send_vgpr_next_b32 v1023, v1023, v1023 sema_wave_id:0 sema_wave_id_refl:0 wait_va_vdst:0 ; encoding: [0x01,0x00,0x00,0xfa,0xff,0xff,0xff,0x3f]

v_send_vgpr_next_b32 v1, v2, v3 sema_id:4 sema_wave_id:5 sema_id_refl:6 sema_wave_id_refl:7 wait_va_vdst:8
// GFX13: v_send_vgpr_next_b32 v1, v2, v3 sema_id:4 sema_wave_id:5 sema_id_refl:6 sema_wave_id_refl:7 wait_va_vdst:8 ; encoding: [0xd1,0x75,0x08,0xfa,0x03,0x04,0x20,0x00]

v_send_vgpr_next_b32 v1, v2, v3 sema_id:-1 sema_wave_id:-2 sema_id_refl:-3 sema_wave_id_refl:-4
// GFX13: v_send_vgpr_next_b32 v1, v2, v3 sema_id:7 sema_wave_id:6 sema_id_refl:5 sema_wave_id_refl:4 wait_va_vdst:0 ; encoding: [0xbd,0x46,0x00,0xfa,0x03,0x04,0x20,0x00]

v_send_vgpr_next_b32 off, off, off
// GFX13: v_send_vgpr_next_b32 off, off, off sema_wave_id:0 sema_wave_id_refl:0 wait_va_vdst:0 ; encoding: [0x00,0x00,0x00,0xfa,0x00,0x00,0x00,0x00]

v_send_vgpr_prev_b32 v1, v2, v3
// GFX13: v_send_vgpr_prev_b32 v1, v2, v3 sema_wave_id:0 sema_wave_id_refl:0 wait_va_vdst:0 ; encoding: [0x01,0x00,0x20,0xfa,0x03,0x04,0x20,0x00]

v_send_vgpr_prev_b32 v1023, v1023, v1023
// GFX13: v_send_vgpr_prev_b32 v1023, v1023, v1023 sema_wave_id:0 sema_wave_id_refl:0 wait_va_vdst:0 ; encoding: [0x01,0x00,0x20,0xfa,0xff,0xff,0xff,0x3f]

v_send_vgpr_prev_b32 v1, v2, v3 sema_id:4 sema_wave_id:5 sema_id_refl:6 sema_wave_id_refl:7 wait_va_vdst:8
// GFX13: v_send_vgpr_prev_b32 v1, v2, v3 sema_id:4 sema_wave_id:5 sema_id_refl:6 sema_wave_id_refl:7 wait_va_vdst:8 ; encoding: [0xd1,0x75,0x28,0xfa,0x03,0x04,0x20,0x00]

v_send_vgpr_prev_b32 v1, v2, v3 sema_id:-1 sema_wave_id:-2 sema_id_refl:-3 sema_wave_id_refl:-4
// GFX13: v_send_vgpr_prev_b32 v1, v2, v3 sema_id:7 sema_wave_id:6 sema_id_refl:5 sema_wave_id_refl:4 wait_va_vdst:0 ; encoding: [0xbd,0x46,0x20,0xfa,0x03,0x04,0x20,0x00]

v_send_vgpr_prev_b32 off, off, off
// GFX13: v_send_vgpr_prev_b32 off, off, off sema_wave_id:0 sema_wave_id_refl:0 wait_va_vdst:0 ; encoding: [0x00,0x00,0x20,0xfa,0x00,0x00,0x00,0x00]
