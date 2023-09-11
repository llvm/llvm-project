// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s | FileCheck --check-prefix=GFX1210 %s

v_mov_b64_e32 v[4:5], v[2:3]
// GFX1210: encoding: [0x02,0x3b,0x08,0x7e]

v_mov_b64 v[4:5], v[254:255]
// GFX1210: encoding: [0xfe,0x3b,0x08,0x7e]

v_mov_b64 v[4:5], s[2:3]
// GFX1210: encoding: [0x02,0x3a,0x08,0x7e]

v_mov_b64 v[4:5], vcc
// GFX1210: encoding: [0x6a,0x3a,0x08,0x7e]

v_mov_b64 v[4:5], exec
// GFX1210: encoding: [0x7e,0x3a,0x08,0x7e]

v_mov_b64 v[4:5], null
// GFX1210: encoding: [0x7c,0x3a,0x08,0x7e]

v_mov_b64 v[4:5], -1
// GFX1210: encoding: [0xc1,0x3a,0x08,0x7e]

v_mov_b64 v[4:5], 0.5
// GFX1210: encoding: [0xf0,0x3a,0x08,0x7e]

v_mov_b64 v[254:255], 0xaf123456
// GFX1210: encoding: [0xff,0x3a,0xfc,0x7f,0x56,0x34,0x12,0xaf]

v_tanh_f32 v5, v1
// GFX1210: v_tanh_f32_e32 v5, v1                   ; encoding: [0x01,0x3d,0x0a,0x7e]

v_tanh_f32 v5, v255
// GFX1210: v_tanh_f32_e32 v5, v255                 ; encoding: [0xff,0x3d,0x0a,0x7e]

v_tanh_f32 v5, s1
// GFX1210: v_tanh_f32_e32 v5, s1                   ; encoding: [0x01,0x3c,0x0a,0x7e]

v_tanh_f32 v5, s105
// GFX1210: v_tanh_f32_e32 v5, s105                 ; encoding: [0x69,0x3c,0x0a,0x7e]

v_tanh_f32 v5, vcc_lo
// GFX1210: v_tanh_f32_e32 v5, vcc_lo               ; encoding: [0x6a,0x3c,0x0a,0x7e]

v_tanh_f32 v5, vcc_hi
// GFX1210: v_tanh_f32_e32 v5, vcc_hi               ; encoding: [0x6b,0x3c,0x0a,0x7e]

v_tanh_f32 v5, ttmp15
// GFX1210: v_tanh_f32_e32 v5, ttmp15               ; encoding: [0x7b,0x3c,0x0a,0x7e]

v_tanh_f32 v5, m0
// GFX1210: v_tanh_f32_e32 v5, m0                   ; encoding: [0x7d,0x3c,0x0a,0x7e]

v_tanh_f32 v5, exec_lo
// GFX1210: v_tanh_f32_e32 v5, exec_lo              ; encoding: [0x7e,0x3c,0x0a,0x7e]

v_tanh_f32 v5, exec_hi
// GFX1210: v_tanh_f32_e32 v5, exec_hi              ; encoding: [0x7f,0x3c,0x0a,0x7e]

v_tanh_f32 v5, null
// GFX1210: v_tanh_f32_e32 v5, null                 ; encoding: [0x7c,0x3c,0x0a,0x7e]

v_tanh_f32 v5, -1
// GFX1210: v_tanh_f32_e32 v5, -1                   ; encoding: [0xc1,0x3c,0x0a,0x7e]

v_tanh_f32 v5, 0.5
// GFX1210: v_tanh_f32_e32 v5, 0.5                  ; encoding: [0xf0,0x3c,0x0a,0x7e]

v_tanh_f32 v5, src_scc
// GFX1210: v_tanh_f32_e32 v5, src_scc              ; encoding: [0xfd,0x3c,0x0a,0x7e]

v_tanh_f32 v255, 0xaf123456
// GFX1210: v_tanh_f32_e32 v255, 0xaf123456         ; encoding: [0xff,0x3c,0xfe,0x7f,0x56,0x34,0x12,0xaf]

v_tanh_f16 v5, v1
// GFX1210: v_tanh_f16_e32 v5, v1                   ; encoding: [0x01,0x3f,0x0a,0x7e]

v_tanh_f16 v5, v127
// GFX1210: v_tanh_f16_e32 v5, v127                 ; encoding: [0x7f,0x3f,0x0a,0x7e]

v_tanh_f16 v5, s1
// GFX1210: v_tanh_f16_e32 v5, s1                   ; encoding: [0x01,0x3e,0x0a,0x7e]

v_tanh_f16 v5, s105
// GFX1210: v_tanh_f16_e32 v5, s105                 ; encoding: [0x69,0x3e,0x0a,0x7e]

v_tanh_f16 v5, vcc_lo
// GFX1210: v_tanh_f16_e32 v5, vcc_lo               ; encoding: [0x6a,0x3e,0x0a,0x7e]

v_tanh_f16 v5, vcc_hi
// GFX1210: v_tanh_f16_e32 v5, vcc_hi               ; encoding: [0x6b,0x3e,0x0a,0x7e]

v_tanh_f16 v5, ttmp15
// GFX1210: v_tanh_f16_e32 v5, ttmp15               ; encoding: [0x7b,0x3e,0x0a,0x7e]

v_tanh_f16 v5, m0
// GFX1210: v_tanh_f16_e32 v5, m0                   ; encoding: [0x7d,0x3e,0x0a,0x7e]

v_tanh_f16 v5, exec_lo
// GFX1210: v_tanh_f16_e32 v5, exec_lo              ; encoding: [0x7e,0x3e,0x0a,0x7e]

v_tanh_f16 v5, exec_hi
// GFX1210: v_tanh_f16_e32 v5, exec_hi              ; encoding: [0x7f,0x3e,0x0a,0x7e]

v_tanh_f16 v5, null
// GFX1210: v_tanh_f16_e32 v5, null                 ; encoding: [0x7c,0x3e,0x0a,0x7e]

v_tanh_f16 v5, -1
// GFX1210: v_tanh_f16_e32 v5, -1                   ; encoding: [0xc1,0x3e,0x0a,0x7e]

v_tanh_f16 v5, 0.5
// GFX1210: v_tanh_f16_e32 v5, 0.5                  ; encoding: [0xf0,0x3e,0x0a,0x7e]

v_tanh_f16 v5, src_scc
// GFX1210: v_tanh_f16_e32 v5, src_scc              ; encoding: [0xfd,0x3e,0x0a,0x7e]

v_tanh_f16 v127, 0x8000
// GFX1210: v_tanh_f16_e32 v127, 0x8000             ; encoding: [0xff,0x3e,0xfe,0x7e,0x00,0x80,0x00,0x00]

v_tanh_bf16 v5, v1
// GFX1210: v_tanh_bf16_e32 v5, v1                  ; encoding: [0x01,0x95,0x0a,0x7e]

v_tanh_bf16 v5, v127
// GFX1210: v_tanh_bf16_e32 v5, v127                ; encoding: [0x7f,0x95,0x0a,0x7e]

v_tanh_bf16 v5, s1
// GFX1210: v_tanh_bf16_e32 v5, s1                  ; encoding: [0x01,0x94,0x0a,0x7e]

v_tanh_bf16 v5, s105
// GFX1210: v_tanh_bf16_e32 v5, s105                ; encoding: [0x69,0x94,0x0a,0x7e]

v_tanh_bf16 v5, vcc_lo
// GFX1210: v_tanh_bf16_e32 v5, vcc_lo              ; encoding: [0x6a,0x94,0x0a,0x7e]

v_tanh_bf16 v5, vcc_hi
// GFX1210: v_tanh_bf16_e32 v5, vcc_hi              ; encoding: [0x6b,0x94,0x0a,0x7e]

v_tanh_bf16 v5, ttmp15
// GFX1210: v_tanh_bf16_e32 v5, ttmp15              ; encoding: [0x7b,0x94,0x0a,0x7e]

v_tanh_bf16 v5, m0
// GFX1210: v_tanh_bf16_e32 v5, m0                  ; encoding: [0x7d,0x94,0x0a,0x7e]

v_tanh_bf16 v5, exec_lo
// GFX1210: v_tanh_bf16_e32 v5, exec_lo             ; encoding: [0x7e,0x94,0x0a,0x7e]

v_tanh_bf16 v5, exec_hi
// GFX1210: v_tanh_bf16_e32 v5, exec_hi             ; encoding: [0x7f,0x94,0x0a,0x7e]

v_tanh_bf16 v5, null
// GFX1210: v_tanh_bf16_e32 v5, null                ; encoding: [0x7c,0x94,0x0a,0x7e]

v_tanh_bf16 v5, -1
// GFX1210: v_tanh_bf16_e32 v5, -1                  ; encoding: [0xc1,0x94,0x0a,0x7e]

v_tanh_bf16 v5, 0.5
// GFX1210: v_tanh_bf16_e32 v5, 0.5                 ; encoding: [0xf0,0x94,0x0a,0x7e]

v_tanh_bf16 v5, src_scc
// GFX1210: v_tanh_bf16_e32 v5, src_scc             ; encoding: [0xfd,0x94,0x0a,0x7e]

v_tanh_bf16 v127, 0x8000
// GFX1210: v_tanh_bf16_e32 v127, 0x8000            ; encoding: [0xff,0x94,0xfe,0x7e,0x00,0x80,0x00,0x00]
