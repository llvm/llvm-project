// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -show-encoding %s | FileCheck --check-prefix=GFX1250 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=WAVESIZE-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1250 %s 2>&1 | FileCheck --check-prefix=GFX1250-ERR --implicit-check-not=error: %s

global_load_tr8_b64 v[2:3], v0, s[0:1]
// GFX1250: global_load_tr8_b64 v[2:3], v0, s[0:1]  ; encoding: [0x00,0x00,0x16,0xee,0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr8_b64 v[2:3], v0, s[0:1] offset:64
// GFX1250: global_load_tr8_b64 v[2:3], v0, s[0:1] offset:64 ; encoding: [0x00,0x00,0x16,0xee,0x02,0x00,0x00,0x00,0x00,0x40,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr8_b64 v[2:3], v0, s[0:1] offset:-64
// GFX1250: global_load_tr8_b64 v[2:3], v0, s[0:1] offset:-64 ; encoding: [0x00,0x00,0x16,0xee,0x02,0x00,0x00,0x00,0x00,0xc0,0xff,0xff]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr8_b64 v[2:3], v[4:5], off
// GFX1250: global_load_tr8_b64 v[2:3], v[4:5], off ; encoding: [0x7c,0x00,0x16,0xee,0x02,0x00,0x00,0x00,0x04,0x00,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr8_b64 v[2:3], v[4:5], off offset:64
// GFX1250: global_load_tr8_b64 v[2:3], v[4:5], off offset:64 ; encoding: [0x7c,0x00,0x16,0xee,0x02,0x00,0x00,0x00,0x04,0x40,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr8_b64 v[2:3], v[4:5], off offset:-64
// GFX1250: global_load_tr8_b64 v[2:3], v[4:5], off offset:-64 ; encoding: [0x7c,0x00,0x16,0xee,0x02,0x00,0x00,0x00,0x04,0xc0,0xff,0xff]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr8_b64 v1, v0, s[0:1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr8_b64 v[2:3], s[3:4], off
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register alignment
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr4_b64 v[2:3], v0, s[0:1]
// GFX1250: global_load_tr4_b64 v[2:3], v0, s[0:1]  ; encoding: [0x00,0xc0,0x1c,0xee,0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr4_b64 v[2:3], v0, s[0:1] offset:64
// GFX1250: global_load_tr4_b64 v[2:3], v0, s[0:1] offset:64 ; encoding: [0x00,0xc0,0x1c,0xee,0x02,0x00,0x00,0x00,0x00,0x40,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr4_b64 v[2:3], v0, s[0:1] offset:-64
// GFX1250: global_load_tr4_b64 v[2:3], v0, s[0:1] offset:-64 ; encoding: [0x00,0xc0,0x1c,0xee,0x02,0x00,0x00,0x00,0x00,0xc0,0xff,0xff]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr4_b64 v[2:3], v[4:5], off
// GFX1250: global_load_tr4_b64 v[2:3], v[4:5], off ; encoding: [0x7c,0xc0,0x1c,0xee,0x02,0x00,0x00,0x00,0x04,0x00,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr4_b64 v[2:3], v[4:5], off offset:64
// GFX1250: global_load_tr4_b64 v[2:3], v[4:5], off offset:64 ; encoding: [0x7c,0xc0,0x1c,0xee,0x02,0x00,0x00,0x00,0x04,0x40,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr4_b64 v[2:3], v[4:5], off offset:-64
// GFX1250: global_load_tr4_b64 v[2:3], v[4:5], off offset:-64 ; encoding: [0x7c,0xc0,0x1c,0xee,0x02,0x00,0x00,0x00,0x04,0xc0,0xff,0xff]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr4_b64 v1, v0, s[0:1]
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr4_b64 v[2:3], s[3:4], off
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register alignment
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr16_b128 v[2:5], v0, s[0:1]
// GFX1250: global_load_tr16_b128 v[2:5], v0, s[0:1] ; encoding: [0x00,0xc0,0x15,0xee,0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr16_b128 v[2:5], v0, s[0:1] offset:64
// GFX1250: global_load_tr16_b128 v[2:5], v0, s[0:1] offset:64 ; encoding: [0x00,0xc0,0x15,0xee,0x02,0x00,0x00,0x00,0x00,0x40,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr16_b128 v[2:5], v0, s[0:1] offset:-64
// GFX1250: global_load_tr16_b128 v[2:5], v0, s[0:1] offset:-64 ; encoding: [0x00,0xc0,0x15,0xee,0x02,0x00,0x00,0x00,0x00,0xc0,0xff,0xff]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr16_b128 v[2:5], v[6:7], off
// GFX1250: global_load_tr16_b128 v[2:5], v[6:7], off ; encoding: [0x7c,0xc0,0x15,0xee,0x02,0x00,0x00,0x00,0x06,0x00,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr16_b128 v[2:5], v[6:7], off offset:64
// GFX1250: global_load_tr16_b128 v[2:5], v[6:7], off offset:64 ; encoding: [0x7c,0xc0,0x15,0xee,0x02,0x00,0x00,0x00,0x06,0x40,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr16_b128 v[2:5], v[6:7], off offset:-64
// GFX1250: global_load_tr16_b128 v[2:5], v[6:7], off offset:-64 ; encoding: [0x7c,0xc0,0x15,0xee,0x02,0x00,0x00,0x00,0x06,0xc0,0xff,0xff]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr16_b128 v[2:3], v[6:7], off
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr16_b128 v[2:5], s[5:6], off
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register alignment
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr6_b96 v[2:4], v0, s[0:1]
// GFX1250: global_load_tr6_b96 v[2:4], v0, s[0:1]  ; encoding: [0x00,0x00,0x1d,0xee,0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr6_b96 v[3:5], v0, s[0:1]
// GFX1250: global_load_tr6_b96 v[3:5], v0, s[0:1]  ; encoding: [0x00,0x00,0x1d,0xee,0x03,0x00,0x00,0x00,0x00,0x00,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr6_b96 v[2:4], v0, s[0:1] offset:64
// GFX1250: global_load_tr6_b96 v[2:4], v0, s[0:1] offset:64 ; encoding: [0x00,0x00,0x1d,0xee,0x02,0x00,0x00,0x00,0x00,0x40,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr6_b96 v[2:4], v0, s[0:1] offset:-64
// GFX1250: global_load_tr6_b96 v[2:4], v0, s[0:1] offset:-64 ; encoding: [0x00,0x00,0x1d,0xee,0x02,0x00,0x00,0x00,0x00,0xc0,0xff,0xff]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr6_b96 v[2:4], v[6:7], off
// GFX1250: global_load_tr6_b96 v[2:4], v[6:7], off ; encoding: [0x7c,0x00,0x1d,0xee,0x02,0x00,0x00,0x00,0x06,0x00,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr6_b96 v[2:4], v[6:7], off offset:64
// GFX1250: global_load_tr6_b96 v[2:4], v[6:7], off offset:64 ; encoding: [0x7c,0x00,0x1d,0xee,0x02,0x00,0x00,0x00,0x06,0x40,0x00,0x00]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr6_b96 v[2:4], v[6:7], off offset:-64
// GFX1250: global_load_tr6_b96 v[2:4], v[6:7], off offset:-64 ; encoding: [0x7c,0x00,0x1d,0xee,0x02,0x00,0x00,0x00,0x06,0xc0,0xff,0xff]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr6_b96 v[2:3], v[6:7], off
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

global_load_tr6_b96 v[2:5], s[5:6], off
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid register alignment
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr8_b64 v[2:3], v0
// GFX1250: ds_load_tr8_b64 v[2:3], v0              ; encoding: [0x00,0x00,0xf4,0xdb,0x00,0x00,0x00,0x02]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr8_b64 v[2:3], v0 offset:64
// GFX1250: ds_load_tr8_b64 v[2:3], v0 offset:64    ; encoding: [0x40,0x00,0xf4,0xdb,0x00,0x00,0x00,0x02]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr8_b64 v[2:3], v0 offset:-64
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr8_b64 v1, v0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr8_b64 v[2:3], s0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr4_b64 v[2:3], v0
// GFX1250: ds_load_tr4_b64 v[2:3], v0              ; encoding: [0x00,0x00,0xe8,0xdb,0x00,0x00,0x00,0x02]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr4_b64 v[2:3], v0 offset:64
// GFX1250: ds_load_tr4_b64 v[2:3], v0 offset:64    ; encoding: [0x40,0x00,0xe8,0xdb,0x00,0x00,0x00,0x02]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr4_b64 v[2:3], v0 offset:-64
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr4_b64 v1, v0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr4_b64 v[2:3], s0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr16_b128 v[2:5], v0
// GFX1250: ds_load_tr16_b128 v[2:5], v0            ; encoding: [0x00,0x00,0xf0,0xdb,0x00,0x00,0x00,0x02]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr16_b128 v[2:5], v0 offset:64
// GFX1250: ds_load_tr16_b128 v[2:5], v0 offset:64  ; encoding: [0x40,0x00,0xf0,0xdb,0x00,0x00,0x00,0x02]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr16_b128 v[2:5], v0 offset:-64
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr16_b128 v[2:3], v0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr16_b128 v[2:5], s0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr6_b96 v[2:4], v0
// GFX1250: ds_load_tr6_b96 v[2:4], v0              ; encoding: [0x00,0x00,0xec,0xdb,0x00,0x00,0x00,0x02]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr6_b96 v[3:5], v0
// GFX1250: ds_load_tr6_b96 v[3:5], v0              ; encoding: [0x00,0x00,0xec,0xdb,0x00,0x00,0x00,0x03]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr6_b96 v[2:4], v0 offset:64
// GFX1250: ds_load_tr6_b96 v[2:4], v0 offset:64    ; encoding: [0x40,0x00,0xec,0xdb,0x00,0x00,0x00,0x02]
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr6_b96 v[2:4], v0 offset:-64
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 16-bit unsigned offset
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr6_b96 v[2:3], v0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32

ds_load_tr6_b96 v[2:4], s0
// GFX1250-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// WAVESIZE-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction requires wavesize=32
