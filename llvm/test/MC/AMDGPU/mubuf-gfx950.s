// RUN: llvm-mc -triple=amdgcn -mcpu=gfx950 -show-encoding %s | FileCheck -check-prefix=GFX950 %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx942 -show-encoding %s 2>&1 | FileCheck -check-prefix=ERR %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx803 -show-encoding %s 2>&1 | FileCheck -check-prefix=ERR %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1030 -show-encoding %s 2>&1 | FileCheck -check-prefix=ERR %s

// FIXME: Bad diagnostics on unsupported subtarget

// GFX950: buffer_load_dwordx3 off, s[8:11], s3 lds ; encoding: [0x00,0x00,0x59,0xe0,0x00,0x00,0x02,0x03]
// ERR: :[[@LINE+1]]:21: error: invalid operand for instruction
buffer_load_dwordx3 off, s[8:11], s3 lds

// GFX950: buffer_load_dwordx3 off, s[8:11], s3 offset:4095 lds ; encoding: [0xff,0x0f,0x59,0xe0,0x00,0x00,0x02,0x03]
// ERR: :[[@LINE+1]]:38: error: not a valid operand
buffer_load_dwordx3 off, s[8:11], s3 offset:4095 lds

// GFX950: buffer_load_dwordx3 v0, s[8:11], s101 offen lds ; encoding: [0x00,0x10,0x59,0xe0,0x00,0x00,0x02,0x65]
// ERR: :[[@LINE+1]]:39: error: invalid operand for instruction
buffer_load_dwordx3 v0, s[8:11], s101 offen lds



// GFX950: buffer_load_dwordx4 off, s[8:11], s3 lds ; encoding: [0x00,0x00,0x5d,0xe0,0x00,0x00,0x02,0x03]
// ERR: :[[@LINE+1]]:21: error: invalid operand for instruction
buffer_load_dwordx4 off, s[8:11], s3 lds

// GFX950: buffer_load_dwordx4 off, s[8:11], s3 offset:4095 lds ; encoding: [0xff,0x0f,0x5d,0xe0,0x00,0x00,0x02,0x03]
// ERR: :[[@LINE+1]]:38: error: not a valid operand
buffer_load_dwordx4 off, s[8:11], s3 offset:4095 lds

// GFX950: buffer_load_dwordx4 v0, s[8:11], s101 offen lds ; encoding: [0x00,0x10,0x5d,0xe0,0x00,0x00,0x02,0x65]
// ERR: :[[@LINE+1]]:39: error: invalid operand for instruction
buffer_load_dwordx4 v0, s[8:11], s101 offen lds
