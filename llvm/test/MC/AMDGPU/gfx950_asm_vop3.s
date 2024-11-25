// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx906 -show-encoding %s 2>&1 | FileCheck -check-prefix=GFX906-ERR %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx940 -show-encoding %s 2>&1 | FileCheck -check-prefix=GFX940-ERR %s
// RUN: llvm-mc -arch=amdgcn -mcpu=gfx950 -show-encoding < %s | FileCheck --check-prefix=GFX950 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck -check-prefix=GFX12-ERR %s

v_cvt_pk_bf16_f32 v5, v1, v2
// GFX906-ERR: error: instruction not supported on this GPU
// GFX940-ERR: error: instruction not supported on this GPU
// GFX950: v_cvt_pk_bf16_f32 v5, v1, v2            ; encoding: [0x05,0x00,0x68,0xd2,0x01,0x05,0x02,0x00]
// GFX12-ERR: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32 v5, v255, v255
// GFX906-ERR: error: instruction not supported on this GPU
// GFX940-ERR: error: instruction not supported on this GPU
// GFX950: v_cvt_pk_bf16_f32 v5, v255, v255        ; encoding: [0x05,0x00,0x68,0xd2,0xff,0xff,0x03,0x00]
// GFX12-ERR: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32 v5, v1, s2
// GFX906-ERR: error: instruction not supported on this GPU
// GFX940-ERR: error: instruction not supported on this GPU
// GFX950: v_cvt_pk_bf16_f32 v5, v1, s2           ; encoding: [0x05,0x00,0x68,0xd2,0x01,0x05,0x00,0x00]
// GFX12-ERR: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32 v5, m0, 0.5
// GFX906-ERR: error: instruction not supported on this GPU
// GFX940-ERR: error: instruction not supported on this GPU
// GFX950: v_cvt_pk_bf16_f32 v5, m0, 0.5           ; encoding: [0x05,0x00,0x68,0xd2,0x7c,0xe0,0x01,0x00]
// GFX12-ERR: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32 v5, -1, exec_hi
// GFX906-ERR: error: instruction not supported on this GPU
// GFX940-ERR: error: instruction not supported on this GPU
// GFX950: v_cvt_pk_bf16_f32 v5, -1, exec_hi       ; encoding: [0x05,0x00,0x68,0xd2,0xc1,0xfe,0x00,0x00]
// GFX12-ERR: error: instruction not supported on this GPU

v_cvt_pk_bf16_f32 v5, 0.5, m0 mul:2
// GFX906-ERR: error: instruction not supported on this GPU
// GFX940-ERR: error: instruction not supported on this GPU
// GFX950: v_cvt_pk_bf16_f32 v5, 0.5, m0 mul:2     ; encoding: [0x05,0x00,0x68,0xd2,0xf0,0xf8,0x00,0x08]
// GFX12-ERR: error: instruction not supported on this GPU

v_bitop3_b32 v5, v1, v2, s3
// GFX906-ERR: error: instruction not supported on this GPU
// GFX940-ERR: error: instruction not supported on this GPU
// GFX950: v_bitop3_b32 v5, v1, v2, s3             ; encoding: [0x05,0x00,0x34,0xd2,0x01,0x05,0x0e,0x00]
// GFX12-ERR: error: instruction not supported on this GPU

v_bitop3_b32 v5, v1, v2, s3 bitop3:161
// GFX906-ERR: error: instruction not supported on this GPU
// GFX940-ERR: error: instruction not supported on this GPU
// GFX950: v_bitop3_b32 v5, v1, v2, s3 bitop3:0xa1 ; encoding: [0x05,0x04,0x34,0xd2,0x01,0x05,0x0e,0x30]
// GFX12-ERR: error: instruction not supported on this GPU

v_bitop3_b32 v5, m0, 0.5, m0 bitop3:5
// GFX906-ERR: error: instruction not supported on this GPU
// GFX940-ERR: error: instruction not supported on this GPU
// GFX950: v_bitop3_b32 v5, m0, 0.5, m0 bitop3:5   ; encoding: [0x05,0x00,0x34,0xd2,0x7c,0xe0,0xf1,0xa1]
// GFX12-ERR: error: instruction not supported on this GPU

v_bitop3_b32 v5, 0.5, m0, 0.5 bitop3:101
// GFX906-ERR: error: instruction not supported on this GPU
// GFX940-ERR: error: instruction not supported on this GPU
// GFX950: v_bitop3_b32 v5, 0.5, m0, 0.5 bitop3:0x65 ; encoding: [0x05,0x04,0x34,0xd2,0xf0,0xf8,0xc0,0xab]
// GFX12-ERR: error: instruction not supported on this GPU

v_bitop3_b16 v5, v1, v2, s3
// GFX906-ERR: error: instruction not supported on this GPU
// GFX940-ERR: error: instruction not supported on this GPU
// GFX950: v_bitop3_b16 v5, v1, v2, s3             ; encoding: [0x05,0x00,0x33,0xd2,0x01,0x05,0x0e,0x00]
// GFX12-ERR: error: instruction not supported on this GPU

v_bitop3_b16 v5, v1, v2, s3 bitop3:161
// GFX906-ERR: error: instruction not supported on this GPU
// GFX940-ERR: error: instruction not supported on this GPU
// GFX950: v_bitop3_b16 v5, v1, v2, s3 bitop3:0xa1 ; encoding: [0x05,0x04,0x33,0xd2,0x01,0x05,0x0e,0x30]
// GFX12-ERR: error: instruction not supported on this GPU
