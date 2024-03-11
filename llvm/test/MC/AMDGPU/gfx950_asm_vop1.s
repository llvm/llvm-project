// RUN: llvm-mc -triple=amdgcn -mcpu=gfx950 -show-encoding %s | FileCheck --check-prefix=GFX950 %s

v_prng_b32 v5, v1
// GFX950: v_prng_b32_e32 v5, v1                   ; encoding: [0x01,0xb1,0x0a,0x7e]

v_prng_b32 v5, v255
// GFX950: v_prng_b32_e32 v5, v255                 ; encoding: [0xff,0xb1,0x0a,0x7e]

v_prng_b32 v5, s1
// GFX950: v_prng_b32_e32 v5, s1                   ; encoding: [0x01,0xb0,0x0a,0x7e]

v_prng_b32 v5, s101
// GFX950: v_prng_b32_e32 v5, s101                 ; encoding: [0x65,0xb0,0x0a,0x7e]

v_prng_b32 v5, vcc_lo
// GFX950: v_prng_b32_e32 v5, vcc_lo               ; encoding: [0x6a,0xb0,0x0a,0x7e]

v_prng_b32 v5, vcc_hi
// GFX950: v_prng_b32_e32 v5, vcc_hi               ; encoding: [0x6b,0xb0,0x0a,0x7e]

v_prng_b32 v5, ttmp15
// GFX950: v_prng_b32_e32 v5, ttmp15               ; encoding: [0x7b,0xb0,0x0a,0x7e]

v_prng_b32 v5, m0
// GFX950: v_prng_b32_e32 v5, m0                   ; encoding: [0x7c,0xb0,0x0a,0x7e]

v_prng_b32 v5, exec_lo
// GFX950: v_prng_b32_e32 v5, exec_lo              ; encoding: [0x7e,0xb0,0x0a,0x7e]

v_prng_b32 v5, exec_hi
// GFX950: v_prng_b32_e32 v5, exec_hi              ; encoding: [0x7f,0xb0,0x0a,0x7e]

v_prng_b32 v5, -1
// GFX950: v_prng_b32_e32 v5, -1                   ; encoding: [0xc1,0xb0,0x0a,0x7e]

v_prng_b32 v5, 0.5
// GFX950: v_prng_b32_e32 v5, 0.5                  ; encoding: [0xf0,0xb0,0x0a,0x7e]

v_prng_b32 v5, src_scc
// GFX950: v_prng_b32_e32 v5, src_scc              ; encoding: [0xfd,0xb0,0x0a,0x7e]

v_prng_b32 v255, 0xaf123456
// GFX950: v_prng_b32_e32 v255, 0xaf123456         ; encoding: [0xff,0xb0,0xfe,0x7f,0x56,0x34,0x12,0xaf]
