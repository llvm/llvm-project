// RUN: llvm-mc -triple=amdgcn -mcpu=gfx950 -show-encoding %s | FileCheck --check-prefix=GFX950 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx940 %s 2>&1 | FileCheck -check-prefix=GFX940-ERR --strict-whitespace  %s

v_prng_b32 v5, v1
// GFX950: v_prng_b32_e32 v5, v1                   ; encoding: [0x01,0xb1,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_prng_b32 v5, v255
// GFX950: v_prng_b32_e32 v5, v255                 ; encoding: [0xff,0xb1,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_prng_b32 v5, s1
// GFX950: v_prng_b32_e32 v5, s1                   ; encoding: [0x01,0xb0,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_prng_b32 v5, s101
// GFX950: v_prng_b32_e32 v5, s101                 ; encoding: [0x65,0xb0,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_prng_b32 v5, vcc_lo
// GFX950: v_prng_b32_e32 v5, vcc_lo               ; encoding: [0x6a,0xb0,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_prng_b32 v5, vcc_hi
// GFX950: v_prng_b32_e32 v5, vcc_hi               ; encoding: [0x6b,0xb0,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_prng_b32 v5, ttmp15
// GFX950: v_prng_b32_e32 v5, ttmp15               ; encoding: [0x7b,0xb0,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_prng_b32 v5, m0
// GFX950: v_prng_b32_e32 v5, m0                   ; encoding: [0x7c,0xb0,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_prng_b32 v5, exec_lo
// GFX950: v_prng_b32_e32 v5, exec_lo              ; encoding: [0x7e,0xb0,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_prng_b32 v5, exec_hi
// GFX950: v_prng_b32_e32 v5, exec_hi              ; encoding: [0x7f,0xb0,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_prng_b32 v5, -1
// GFX950: v_prng_b32_e32 v5, -1                   ; encoding: [0xc1,0xb0,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_prng_b32 v5, 0.5
// GFX950: v_prng_b32_e32 v5, 0.5                  ; encoding: [0xf0,0xb0,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_prng_b32 v5, src_scc
// GFX950: v_prng_b32_e32 v5, src_scc              ; encoding: [0xfd,0xb0,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_prng_b32 v255, 0xaf123456
// GFX950: v_prng_b32_e32 v255, 0xaf123456         ; encoding: [0xff,0xb0,0xfe,0x7f,0x56,0x34,0x12,0xaf]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v5, v1
// GFX950: v_cvt_f32_bf16_e32 v5, v1               ; encoding: [0x01,0xb7,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v5, v127
// GFX950: v_cvt_f32_bf16_e32 v5, v127             ; encoding: [0x7f,0xb7,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v5, s1
// GFX950: v_cvt_f32_bf16_e32 v5, s1               ; encoding: [0x01,0xb6,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v5, vcc_lo
// GFX950: v_cvt_f32_bf16_e32 v5, vcc_lo           ; encoding: [0x6a,0xb6,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v5, vcc_hi
// GFX950: v_cvt_f32_bf16_e32 v5, vcc_hi           ; encoding: [0x6b,0xb6,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v5, ttmp15
// GFX950: v_cvt_f32_bf16_e32 v5, ttmp15           ; encoding: [0x7b,0xb6,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v5, m0
// GFX950: v_cvt_f32_bf16_e32 v5, m0               ; encoding: [0x7c,0xb6,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v5, exec_lo
// GFX950: v_cvt_f32_bf16_e32 v5, exec_lo          ; encoding: [0x7e,0xb6,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v5, exec_hi
// GFX950: v_cvt_f32_bf16_e32 v5, exec_hi          ; encoding: [0x7f,0xb6,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v5, -1
// GFX950: v_cvt_f32_bf16_e32 v5, -1               ; encoding: [0xc1,0xb6,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v5, 0.5
// GFX950: v_cvt_f32_bf16_e32 v5, 0.5              ; encoding: [0xf0,0xb6,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v5, src_scc
// GFX950: v_cvt_f32_bf16_e32 v5, src_scc          ; encoding: [0xfd,0xb6,0x0a,0x7e]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v127, 0x8000
// GFX950: v_cvt_f32_bf16_e32 v127, 0x8000         ; encoding: [0xff,0xb6,0xfe,0x7e,0x00,0x80,0x00,0x00]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v5, -v1
// GFX950: v_cvt_f32_bf16_e64 v5, -v1              ; encoding: [0x05,0x00,0x9b,0xd1,0x01,0x01,0x00,0x20]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v5, |v1|
// GFX950: v_cvt_f32_bf16_e64 v5, |v1|             ; encoding: [0x05,0x01,0x9b,0xd1,0x01,0x01,0x00,0x00]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v5, -|v1|
// GFX950: v_cvt_f32_bf16_e64 v5, -|v1|            ; encoding: [0x05,0x01,0x9b,0xd1,0x01,0x01,0x00,0x20]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16 v5, v1 clamp mul:2
// GFX950: v_cvt_f32_bf16_e64 v5, v1 clamp mul:2   ; encoding: [0x05,0x80,0x9b,0xd1,0x01,0x01,0x00,0x08]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_cvt_f32_bf16_e64 v5, v1 clamp div:2
// GFX950: v_cvt_f32_bf16_e64 v5, v1 clamp div:2   ; encoding: [0x05,0x80,0x9b,0xd1,0x01,0x01,0x00,0x18]
// GFX940-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
