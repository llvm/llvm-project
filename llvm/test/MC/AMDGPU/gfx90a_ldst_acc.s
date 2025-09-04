// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx908 %s 2>&1 | FileCheck --check-prefix=NOT-GFX90A --implicit-check-not=error: %s
// RUN: llvm-mc -triple=amdgcn -mcpu=gfx90a -show-encoding %s | FileCheck --check-prefix=GFX90A %s

// GFX90A: flat_load_ubyte a5, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x40,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte a5, v[2:3] offset:4095

// GFX90A: flat_load_ubyte a255, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x40,0xdc,0x02,0x00,0x80,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte a255, v[2:3] offset:4095

// GFX90A: flat_load_ubyte a5, v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x40,0xdc,0xfe,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte a5, v[254:255] offset:4095

// GFX90A: flat_load_ubyte a5, v[2:3]      ; encoding: [0x00,0x00,0x40,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte a5, v[2:3]

// GFX90A: flat_load_ubyte a5, v[2:3]      ; encoding: [0x00,0x00,0x40,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte a5, v[2:3]

// GFX90A: flat_load_ubyte a5, v[2:3] offset:7 ; encoding: [0x07,0x00,0x40,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte a5, v[2:3] offset:7

// GFX90A: flat_load_ubyte a5, v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x41,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte a5, v[2:3] offset:4095 glc

// GFX90A: flat_load_ubyte a5, v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x42,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte a5, v[2:3] offset:4095 slc

// GFX90A: flat_load_sbyte a5, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x44,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte a5, v[2:3] offset:4095

// GFX90A: flat_load_sbyte a255, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x44,0xdc,0x02,0x00,0x80,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte a255, v[2:3] offset:4095

// GFX90A: flat_load_sbyte a5, v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x44,0xdc,0xfe,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte a5, v[254:255] offset:4095

// GFX90A: flat_load_sbyte a5, v[2:3]      ; encoding: [0x00,0x00,0x44,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte a5, v[2:3]

// GFX90A: flat_load_sbyte a5, v[2:3]      ; encoding: [0x00,0x00,0x44,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte a5, v[2:3]

// GFX90A: flat_load_sbyte a5, v[2:3] offset:7 ; encoding: [0x07,0x00,0x44,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte a5, v[2:3] offset:7

// GFX90A: flat_load_sbyte a5, v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x45,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte a5, v[2:3] offset:4095 glc

// GFX90A: flat_load_sbyte a5, v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x46,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte a5, v[2:3] offset:4095 slc

// GFX90A: flat_load_ushort a5, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x48,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ushort a5, v[2:3] offset:4095

// GFX90A: flat_load_ushort a255, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x48,0xdc,0x02,0x00,0x80,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ushort a255, v[2:3] offset:4095

// GFX90A: flat_load_ushort a5, v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x48,0xdc,0xfe,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ushort a5, v[254:255] offset:4095

// GFX90A: flat_load_ushort a5, v[2:3]     ; encoding: [0x00,0x00,0x48,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ushort a5, v[2:3]

// GFX90A: flat_load_ushort a5, v[2:3]     ; encoding: [0x00,0x00,0x48,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ushort a5, v[2:3]

// GFX90A: flat_load_ushort a5, v[2:3] offset:7 ; encoding: [0x07,0x00,0x48,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ushort a5, v[2:3] offset:7

// GFX90A: flat_load_ushort a5, v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x49,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ushort a5, v[2:3] offset:4095 glc

// GFX90A: flat_load_ushort a5, v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x4a,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ushort a5, v[2:3] offset:4095 slc

// GFX90A: flat_load_sshort a5, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x4c,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sshort a5, v[2:3] offset:4095

// GFX90A: flat_load_sshort a255, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x4c,0xdc,0x02,0x00,0x80,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sshort a255, v[2:3] offset:4095

// GFX90A: flat_load_sshort a5, v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x4c,0xdc,0xfe,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sshort a5, v[254:255] offset:4095

// GFX90A: flat_load_sshort a5, v[2:3]     ; encoding: [0x00,0x00,0x4c,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sshort a5, v[2:3]

// GFX90A: flat_load_sshort a5, v[2:3]     ; encoding: [0x00,0x00,0x4c,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sshort a5, v[2:3]

// GFX90A: flat_load_sshort a5, v[2:3] offset:7 ; encoding: [0x07,0x00,0x4c,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sshort a5, v[2:3] offset:7

// GFX90A: flat_load_sshort a5, v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x4d,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sshort a5, v[2:3] offset:4095 glc

// GFX90A: flat_load_sshort a5, v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x4e,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sshort a5, v[2:3] offset:4095 slc

// GFX90A: flat_load_dword a5, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x50,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dword a5, v[2:3] offset:4095

// GFX90A: flat_load_dword a255, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x50,0xdc,0x02,0x00,0x80,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dword a255, v[2:3] offset:4095

// GFX90A: flat_load_dword a5, v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x50,0xdc,0xfe,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dword a5, v[254:255] offset:4095

// GFX90A: flat_load_dword a5, v[2:3]      ; encoding: [0x00,0x00,0x50,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dword a5, v[2:3]

// GFX90A: flat_load_dword a5, v[2:3]      ; encoding: [0x00,0x00,0x50,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dword a5, v[2:3]

// GFX90A: flat_load_dword a5, v[2:3] offset:7 ; encoding: [0x07,0x00,0x50,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dword a5, v[2:3] offset:7

// GFX90A: flat_load_dword a5, v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x51,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dword a5, v[2:3] offset:4095 glc

// GFX90A: flat_load_dword a5, v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x52,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dword a5, v[2:3] offset:4095 slc

// GFX90A: flat_load_dwordx2 a[6:7], v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x54,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx2 a[6:7], v[2:3] offset:4095

// GFX90A: flat_load_dwordx2 a[254:255], v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x54,0xdc,0x02,0x00,0x80,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx2 a[254:255], v[2:3] offset:4095

// GFX90A: flat_load_dwordx2 a[6:7], v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x54,0xdc,0xfe,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx2 a[6:7], v[254:255] offset:4095

// GFX90A: flat_load_dwordx2 a[6:7], v[2:3] ; encoding: [0x00,0x00,0x54,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx2 a[6:7], v[2:3]

// GFX90A: flat_load_dwordx2 a[6:7], v[2:3] ; encoding: [0x00,0x00,0x54,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx2 a[6:7], v[2:3]

// GFX90A: flat_load_dwordx2 a[6:7], v[2:3] offset:7 ; encoding: [0x07,0x00,0x54,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx2 a[6:7], v[2:3] offset:7

// GFX90A: flat_load_dwordx2 a[6:7], v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x55,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx2 a[6:7], v[2:3] offset:4095 glc

// GFX90A: flat_load_dwordx2 a[6:7], v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x56,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx2 a[6:7], v[2:3] offset:4095 slc

// GFX90A: flat_load_dwordx3 a[6:8], v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x58,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx3 a[6:8], v[2:3] offset:4095

// GFX90A: flat_load_dwordx3 a[252:254], v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x58,0xdc,0x02,0x00,0x80,0xfc]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx3 a[252:254], v[2:3] offset:4095

// GFX90A: flat_load_dwordx3 a[6:8], v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x58,0xdc,0xfe,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx3 a[6:8], v[254:255] offset:4095

// GFX90A: flat_load_dwordx3 a[6:8], v[2:3] ; encoding: [0x00,0x00,0x58,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx3 a[6:8], v[2:3]

// GFX90A: flat_load_dwordx3 a[6:8], v[2:3] ; encoding: [0x00,0x00,0x58,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx3 a[6:8], v[2:3]

// GFX90A: flat_load_dwordx3 a[6:8], v[2:3] offset:7 ; encoding: [0x07,0x00,0x58,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx3 a[6:8], v[2:3] offset:7

// GFX90A: flat_load_dwordx3 a[6:8], v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x59,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx3 a[6:8], v[2:3] offset:4095 glc

// GFX90A: flat_load_dwordx3 a[6:8], v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x5a,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx3 a[6:8], v[2:3] offset:4095 slc

// GFX90A: flat_load_dwordx4 a[6:9], v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x5c,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx4 a[6:9], v[2:3] offset:4095

// GFX90A: flat_load_dwordx4 a[252:255], v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x5c,0xdc,0x02,0x00,0x80,0xfc]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx4 a[252:255], v[2:3] offset:4095

// GFX90A: flat_load_dwordx4 a[6:9], v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x5c,0xdc,0xfe,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx4 a[6:9], v[254:255] offset:4095

// GFX90A: flat_load_dwordx4 a[6:9], v[2:3] ; encoding: [0x00,0x00,0x5c,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx4 a[6:9], v[2:3]

// GFX90A: flat_load_dwordx4 a[6:9], v[2:3] ; encoding: [0x00,0x00,0x5c,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx4 a[6:9], v[2:3]

// GFX90A: flat_load_dwordx4 a[6:9], v[2:3] offset:7 ; encoding: [0x07,0x00,0x5c,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx4 a[6:9], v[2:3] offset:7

// GFX90A: flat_load_dwordx4 a[6:9], v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x5d,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx4 a[6:9], v[2:3] offset:4095 glc

// GFX90A: flat_load_dwordx4 a[6:9], v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x5e,0xdc,0x02,0x00,0x80,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_dwordx4 a[6:9], v[2:3] offset:4095 slc

// GFX90A: flat_store_byte v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x60,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_byte v[2:3], a2 offset:4095

// GFX90A: flat_store_byte v[254:255], a2 offset:4095 ; encoding: [0xff,0x0f,0x60,0xdc,0xfe,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_byte v[254:255], a2 offset:4095

// GFX90A: flat_store_byte v[2:3], a255 offset:4095 ; encoding: [0xff,0x0f,0x60,0xdc,0x02,0xff,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_byte v[2:3], a255 offset:4095

// GFX90A: flat_store_byte v[2:3], a2      ; encoding: [0x00,0x00,0x60,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_byte v[2:3], a2

// GFX90A: flat_store_byte v[2:3], a2      ; encoding: [0x00,0x00,0x60,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_byte v[2:3], a2

// GFX90A: flat_store_byte v[2:3], a2 offset:7 ; encoding: [0x07,0x00,0x60,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_byte v[2:3], a2 offset:7

// GFX90A: flat_store_byte v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x61,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_byte v[2:3], a2 offset:4095 glc

// GFX90A: flat_store_byte v[2:3], a2 offset:4095 slc ; encoding: [0xff,0x0f,0x62,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_byte v[2:3], a2 offset:4095 slc

// GFX90A: flat_store_byte_d16_hi v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x64,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_byte_d16_hi v[2:3], a2 offset:4095

// GFX90A: flat_store_byte_d16_hi v[254:255], a2 offset:4095 ; encoding: [0xff,0x0f,0x64,0xdc,0xfe,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_byte_d16_hi v[254:255], a2 offset:4095

// GFX90A: flat_store_byte_d16_hi v[2:3], a255 offset:4095 ; encoding: [0xff,0x0f,0x64,0xdc,0x02,0xff,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_byte_d16_hi v[2:3], a255 offset:4095

// GFX90A: flat_store_byte_d16_hi v[2:3], a2 ; encoding: [0x00,0x00,0x64,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_byte_d16_hi v[2:3], a2

// GFX90A: flat_store_byte_d16_hi v[2:3], a2 ; encoding: [0x00,0x00,0x64,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_byte_d16_hi v[2:3], a2

// GFX90A: flat_store_byte_d16_hi v[2:3], a2 offset:7 ; encoding: [0x07,0x00,0x64,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_byte_d16_hi v[2:3], a2 offset:7

// GFX90A: flat_store_byte_d16_hi v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x65,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_byte_d16_hi v[2:3], a2 offset:4095 glc

// GFX90A: flat_store_byte_d16_hi v[2:3], a2 offset:4095 slc ; encoding: [0xff,0x0f,0x66,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_byte_d16_hi v[2:3], a2 offset:4095 slc

// GFX90A: flat_store_short v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x68,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_short v[2:3], a2 offset:4095

// GFX90A: flat_store_short v[254:255], a2 offset:4095 ; encoding: [0xff,0x0f,0x68,0xdc,0xfe,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_short v[254:255], a2 offset:4095

// GFX90A: flat_store_short v[2:3], a255 offset:4095 ; encoding: [0xff,0x0f,0x68,0xdc,0x02,0xff,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_short v[2:3], a255 offset:4095

// GFX90A: flat_store_short v[2:3], a2     ; encoding: [0x00,0x00,0x68,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_short v[2:3], a2

// GFX90A: flat_store_short v[2:3], a2     ; encoding: [0x00,0x00,0x68,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_short v[2:3], a2

// GFX90A: flat_store_short v[2:3], a2 offset:7 ; encoding: [0x07,0x00,0x68,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_short v[2:3], a2 offset:7

// GFX90A: flat_store_short v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x69,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_short v[2:3], a2 offset:4095 glc

// GFX90A: flat_store_short v[2:3], a2 offset:4095 slc ; encoding: [0xff,0x0f,0x6a,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_short v[2:3], a2 offset:4095 slc

// GFX90A: flat_store_short_d16_hi v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x6c,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_short_d16_hi v[2:3], a2 offset:4095

// GFX90A: flat_store_short_d16_hi v[254:255], a2 offset:4095 ; encoding: [0xff,0x0f,0x6c,0xdc,0xfe,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_short_d16_hi v[254:255], a2 offset:4095

// GFX90A: flat_store_short_d16_hi v[2:3], a255 offset:4095 ; encoding: [0xff,0x0f,0x6c,0xdc,0x02,0xff,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_short_d16_hi v[2:3], a255 offset:4095

// GFX90A: flat_store_short_d16_hi v[2:3], a2 ; encoding: [0x00,0x00,0x6c,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_short_d16_hi v[2:3], a2

// GFX90A: flat_store_short_d16_hi v[2:3], a2 ; encoding: [0x00,0x00,0x6c,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_short_d16_hi v[2:3], a2

// GFX90A: flat_store_short_d16_hi v[2:3], a2 offset:7 ; encoding: [0x07,0x00,0x6c,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_short_d16_hi v[2:3], a2 offset:7

// GFX90A: flat_store_short_d16_hi v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x6d,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_short_d16_hi v[2:3], a2 offset:4095 glc

// GFX90A: flat_store_short_d16_hi v[2:3], a2 offset:4095 slc ; encoding: [0xff,0x0f,0x6e,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_short_d16_hi v[2:3], a2 offset:4095 slc

// GFX90A: flat_store_dword v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x70,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dword v[2:3], a2 offset:4095

// GFX90A: flat_store_dword v[254:255], a2 offset:4095 ; encoding: [0xff,0x0f,0x70,0xdc,0xfe,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dword v[254:255], a2 offset:4095

// GFX90A: flat_store_dword v[2:3], a255 offset:4095 ; encoding: [0xff,0x0f,0x70,0xdc,0x02,0xff,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dword v[2:3], a255 offset:4095

// GFX90A: flat_store_dword v[2:3], a2     ; encoding: [0x00,0x00,0x70,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dword v[2:3], a2

// GFX90A: flat_store_dword v[2:3], a2     ; encoding: [0x00,0x00,0x70,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dword v[2:3], a2

// GFX90A: flat_store_dword v[2:3], a2 offset:7 ; encoding: [0x07,0x00,0x70,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dword v[2:3], a2 offset:7

// GFX90A: flat_store_dword v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x71,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dword v[2:3], a2 offset:4095 glc

// GFX90A: flat_store_dword v[2:3], a2 offset:4095 slc ; encoding: [0xff,0x0f,0x72,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dword v[2:3], a2 offset:4095 slc

// GFX90A: flat_store_dwordx2 v[2:3], a[2:3] offset:4095 ; encoding: [0xff,0x0f,0x74,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx2 v[2:3], a[2:3] offset:4095

// GFX90A: flat_store_dwordx2 v[254:255], a[2:3] offset:4095 ; encoding: [0xff,0x0f,0x74,0xdc,0xfe,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx2 v[254:255], a[2:3] offset:4095

// GFX90A: flat_store_dwordx2 v[2:3], a[254:255] offset:4095 ; encoding: [0xff,0x0f,0x74,0xdc,0x02,0xfe,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx2 v[2:3], a[254:255] offset:4095

// GFX90A: flat_store_dwordx2 v[2:3], a[2:3] ; encoding: [0x00,0x00,0x74,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx2 v[2:3], a[2:3]

// GFX90A: flat_store_dwordx2 v[2:3], a[2:3] ; encoding: [0x00,0x00,0x74,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx2 v[2:3], a[2:3]

// GFX90A: flat_store_dwordx2 v[2:3], a[2:3] offset:7 ; encoding: [0x07,0x00,0x74,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx2 v[2:3], a[2:3] offset:7

// GFX90A: flat_store_dwordx2 v[2:3], a[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x75,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx2 v[2:3], a[2:3] offset:4095 glc

// GFX90A: flat_store_dwordx2 v[2:3], a[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x76,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx2 v[2:3], a[2:3] offset:4095 slc

// GFX90A: flat_store_dwordx3 v[2:3], a[2:4] offset:4095 ; encoding: [0xff,0x0f,0x78,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx3 v[2:3], a[2:4] offset:4095

// GFX90A: flat_store_dwordx3 v[254:255], a[2:4] offset:4095 ; encoding: [0xff,0x0f,0x78,0xdc,0xfe,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx3 v[254:255], a[2:4] offset:4095

// GFX90A: flat_store_dwordx3 v[2:3], a[252:254] offset:4095 ; encoding: [0xff,0x0f,0x78,0xdc,0x02,0xfc,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx3 v[2:3], a[252:254] offset:4095

// GFX90A: flat_store_dwordx3 v[2:3], a[2:4] ; encoding: [0x00,0x00,0x78,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx3 v[2:3], a[2:4]

// GFX90A: flat_store_dwordx3 v[2:3], a[2:4] ; encoding: [0x00,0x00,0x78,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx3 v[2:3], a[2:4]

// GFX90A: flat_store_dwordx3 v[2:3], a[2:4] offset:7 ; encoding: [0x07,0x00,0x78,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx3 v[2:3], a[2:4] offset:7

// GFX90A: flat_store_dwordx3 v[2:3], a[2:4] offset:4095 glc ; encoding: [0xff,0x0f,0x79,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx3 v[2:3], a[2:4] offset:4095 glc

// GFX90A: flat_store_dwordx3 v[2:3], a[2:4] offset:4095 slc ; encoding: [0xff,0x0f,0x7a,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx3 v[2:3], a[2:4] offset:4095 slc

// GFX90A: flat_store_dwordx4 v[2:3], a[2:5] offset:4095 ; encoding: [0xff,0x0f,0x7c,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx4 v[2:3], a[2:5] offset:4095

// GFX90A: flat_store_dwordx4 v[254:255], a[2:5] offset:4095 ; encoding: [0xff,0x0f,0x7c,0xdc,0xfe,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx4 v[254:255], a[2:5] offset:4095

// GFX90A: flat_store_dwordx4 v[2:3], a[252:255] offset:4095 ; encoding: [0xff,0x0f,0x7c,0xdc,0x02,0xfc,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx4 v[2:3], a[252:255] offset:4095

// GFX90A: flat_store_dwordx4 v[2:3], a[2:5] ; encoding: [0x00,0x00,0x7c,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx4 v[2:3], a[2:5]

// GFX90A: flat_store_dwordx4 v[2:3], a[2:5] ; encoding: [0x00,0x00,0x7c,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx4 v[2:3], a[2:5]

// GFX90A: flat_store_dwordx4 v[2:3], a[2:5] offset:7 ; encoding: [0x07,0x00,0x7c,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx4 v[2:3], a[2:5] offset:7

// GFX90A: flat_store_dwordx4 v[2:3], a[2:5] offset:4095 glc ; encoding: [0xff,0x0f,0x7d,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx4 v[2:3], a[2:5] offset:4095 glc

// GFX90A: flat_store_dwordx4 v[2:3], a[2:5] offset:4095 slc ; encoding: [0xff,0x0f,0x7e,0xdc,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_store_dwordx4 v[2:3], a[2:5] offset:4095 slc

// GFX90A: flat_load_ubyte_d16 a5, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x80,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte_d16 a5, v[2:3] offset:4095

// GFX90A: flat_load_ubyte_d16 a255, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x80,0xdc,0x02,0x00,0x80,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte_d16 a255, v[2:3] offset:4095

// GFX90A: flat_load_ubyte_d16 a5, v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x80,0xdc,0xfe,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte_d16 a5, v[254:255] offset:4095

// GFX90A: flat_load_ubyte_d16 a5, v[2:3]  ; encoding: [0x00,0x00,0x80,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte_d16 a5, v[2:3]

// GFX90A: flat_load_ubyte_d16 a5, v[2:3]  ; encoding: [0x00,0x00,0x80,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte_d16 a5, v[2:3]

// GFX90A: flat_load_ubyte_d16 a5, v[2:3] offset:7 ; encoding: [0x07,0x00,0x80,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte_d16 a5, v[2:3] offset:7

// GFX90A: flat_load_ubyte_d16 a5, v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x81,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte_d16 a5, v[2:3] offset:4095 glc

// GFX90A: flat_load_ubyte_d16 a5, v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x82,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte_d16 a5, v[2:3] offset:4095 slc

// GFX90A: flat_load_ubyte_d16_hi a5, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x84,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte_d16_hi a5, v[2:3] offset:4095

// GFX90A: flat_load_ubyte_d16_hi a255, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x84,0xdc,0x02,0x00,0x80,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte_d16_hi a255, v[2:3] offset:4095

// GFX90A: flat_load_ubyte_d16_hi a5, v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x84,0xdc,0xfe,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte_d16_hi a5, v[254:255] offset:4095

// GFX90A: flat_load_ubyte_d16_hi a5, v[2:3] ; encoding: [0x00,0x00,0x84,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte_d16_hi a5, v[2:3]

// GFX90A: flat_load_ubyte_d16_hi a5, v[2:3] ; encoding: [0x00,0x00,0x84,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte_d16_hi a5, v[2:3]

// GFX90A: flat_load_ubyte_d16_hi a5, v[2:3] offset:7 ; encoding: [0x07,0x00,0x84,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte_d16_hi a5, v[2:3] offset:7

// GFX90A: flat_load_ubyte_d16_hi a5, v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x85,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte_d16_hi a5, v[2:3] offset:4095 glc

// GFX90A: flat_load_ubyte_d16_hi a5, v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x86,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_ubyte_d16_hi a5, v[2:3] offset:4095 slc

// GFX90A: flat_load_sbyte_d16 a5, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x88,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte_d16 a5, v[2:3] offset:4095

// GFX90A: flat_load_sbyte_d16 a255, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x88,0xdc,0x02,0x00,0x80,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte_d16 a255, v[2:3] offset:4095

// GFX90A: flat_load_sbyte_d16 a5, v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x88,0xdc,0xfe,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte_d16 a5, v[254:255] offset:4095

// GFX90A: flat_load_sbyte_d16 a5, v[2:3]  ; encoding: [0x00,0x00,0x88,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte_d16 a5, v[2:3]

// GFX90A: flat_load_sbyte_d16 a5, v[2:3]  ; encoding: [0x00,0x00,0x88,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte_d16 a5, v[2:3]

// GFX90A: flat_load_sbyte_d16 a5, v[2:3] offset:7 ; encoding: [0x07,0x00,0x88,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte_d16 a5, v[2:3] offset:7

// GFX90A: flat_load_sbyte_d16 a5, v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x89,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte_d16 a5, v[2:3] offset:4095 glc

// GFX90A: flat_load_sbyte_d16 a5, v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x8a,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte_d16 a5, v[2:3] offset:4095 slc

// GFX90A: flat_load_sbyte_d16_hi a5, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x8c,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte_d16_hi a5, v[2:3] offset:4095

// GFX90A: flat_load_sbyte_d16_hi a255, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x8c,0xdc,0x02,0x00,0x80,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte_d16_hi a255, v[2:3] offset:4095

// GFX90A: flat_load_sbyte_d16_hi a5, v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x8c,0xdc,0xfe,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte_d16_hi a5, v[254:255] offset:4095

// GFX90A: flat_load_sbyte_d16_hi a5, v[2:3] ; encoding: [0x00,0x00,0x8c,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte_d16_hi a5, v[2:3]

// GFX90A: flat_load_sbyte_d16_hi a5, v[2:3] ; encoding: [0x00,0x00,0x8c,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte_d16_hi a5, v[2:3]

// GFX90A: flat_load_sbyte_d16_hi a5, v[2:3] offset:7 ; encoding: [0x07,0x00,0x8c,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte_d16_hi a5, v[2:3] offset:7

// GFX90A: flat_load_sbyte_d16_hi a5, v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x8d,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte_d16_hi a5, v[2:3] offset:4095 glc

// GFX90A: flat_load_sbyte_d16_hi a5, v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x8e,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_sbyte_d16_hi a5, v[2:3] offset:4095 slc

// GFX90A: flat_load_short_d16 a5, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x90,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_short_d16 a5, v[2:3] offset:4095

// GFX90A: flat_load_short_d16 a255, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x90,0xdc,0x02,0x00,0x80,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_short_d16 a255, v[2:3] offset:4095

// GFX90A: flat_load_short_d16 a5, v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x90,0xdc,0xfe,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_short_d16 a5, v[254:255] offset:4095

// GFX90A: flat_load_short_d16 a5, v[2:3]  ; encoding: [0x00,0x00,0x90,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_short_d16 a5, v[2:3]

// GFX90A: flat_load_short_d16 a5, v[2:3]  ; encoding: [0x00,0x00,0x90,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_short_d16 a5, v[2:3]

// GFX90A: flat_load_short_d16 a5, v[2:3] offset:7 ; encoding: [0x07,0x00,0x90,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_short_d16 a5, v[2:3] offset:7

// GFX90A: flat_load_short_d16 a5, v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x91,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_short_d16 a5, v[2:3] offset:4095 glc

// GFX90A: flat_load_short_d16 a5, v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x92,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_short_d16 a5, v[2:3] offset:4095 slc

// GFX90A: flat_load_short_d16_hi a5, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x94,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_short_d16_hi a5, v[2:3] offset:4095

// GFX90A: flat_load_short_d16_hi a255, v[2:3] offset:4095 ; encoding: [0xff,0x0f,0x94,0xdc,0x02,0x00,0x80,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_short_d16_hi a255, v[2:3] offset:4095

// GFX90A: flat_load_short_d16_hi a5, v[254:255] offset:4095 ; encoding: [0xff,0x0f,0x94,0xdc,0xfe,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_short_d16_hi a5, v[254:255] offset:4095

// GFX90A: flat_load_short_d16_hi a5, v[2:3] ; encoding: [0x00,0x00,0x94,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_short_d16_hi a5, v[2:3]

// GFX90A: flat_load_short_d16_hi a5, v[2:3] ; encoding: [0x00,0x00,0x94,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_short_d16_hi a5, v[2:3]

// GFX90A: flat_load_short_d16_hi a5, v[2:3] offset:7 ; encoding: [0x07,0x00,0x94,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_short_d16_hi a5, v[2:3] offset:7

// GFX90A: flat_load_short_d16_hi a5, v[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x95,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_short_d16_hi a5, v[2:3] offset:4095 glc

// GFX90A: flat_load_short_d16_hi a5, v[2:3] offset:4095 slc ; encoding: [0xff,0x0f,0x96,0xdc,0x02,0x00,0x80,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_load_short_d16_hi a5, v[2:3] offset:4095 slc

// GFX90A: flat_atomic_swap a0, v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x01,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_swap a0, v[2:3], a2 offset:4095 glc

// GFX90A: flat_atomic_cmpswap a0, v[2:3], a[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x05,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_cmpswap a0, v[2:3], a[2:3] offset:4095 glc

// GFX90A: flat_atomic_add a0, v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x09,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_add a0, v[2:3], a2 offset:4095 glc

// GFX90A: flat_atomic_sub a0, v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x0d,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_sub a0, v[2:3], a2 offset:4095 glc

// GFX90A: flat_atomic_smin a0, v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x11,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_smin a0, v[2:3], a2 offset:4095 glc

// GFX90A: flat_atomic_umin a0, v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x15,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_umin a0, v[2:3], a2 offset:4095 glc

// GFX90A: flat_atomic_smax a0, v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x19,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_smax a0, v[2:3], a2 offset:4095 glc

// GFX90A: flat_atomic_umax a0, v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x1d,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_umax a0, v[2:3], a2 offset:4095 glc

// GFX90A: flat_atomic_and a0, v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x21,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_and a0, v[2:3], a2 offset:4095 glc

// GFX90A: flat_atomic_or a0, v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x25,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_or a0, v[2:3], a2 offset:4095 glc

// GFX90A: flat_atomic_xor a0, v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x29,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_xor a0, v[2:3], a2 offset:4095 glc

// GFX90A: flat_atomic_inc a0, v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x2d,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_inc a0, v[2:3], a2 offset:4095 glc

// GFX90A: flat_atomic_dec a0, v[2:3], a2 offset:4095 glc ; encoding: [0xff,0x0f,0x31,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_dec a0, v[2:3], a2 offset:4095 glc

// GFX90A: flat_atomic_swap_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x81,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_swap_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc

// GFX90A: flat_atomic_cmpswap_x2 a[0:1], v[2:3], a[2:5] offset:4095 glc ; encoding: [0xff,0x0f,0x85,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_cmpswap_x2 a[0:1], v[2:3], a[2:5] offset:4095 glc

// GFX90A: flat_atomic_add_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x89,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_add_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc

// GFX90A: flat_atomic_sub_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x8d,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_sub_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc

// GFX90A: flat_atomic_smin_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x91,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_smin_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc

// GFX90A: flat_atomic_umin_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x95,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_umin_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc

// GFX90A: flat_atomic_smax_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x99,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_smax_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc

// GFX90A: flat_atomic_umax_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0x9d,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_umax_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc

// GFX90A: flat_atomic_and_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0xa1,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_and_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc

// GFX90A: flat_atomic_or_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0xa5,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_or_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc

// GFX90A: flat_atomic_xor_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0xa9,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_xor_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc

// GFX90A: flat_atomic_inc_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0xad,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_inc_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc

// GFX90A: flat_atomic_dec_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc ; encoding: [0xff,0x0f,0xb1,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_dec_x2 a[0:1], v[2:3], a[2:3] offset:4095 glc

// GFX90A: flat_atomic_swap v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x00,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_swap v[2:3], a2 offset:4095

// GFX90A: flat_atomic_cmpswap v[2:3], a[2:3] offset:4095 ; encoding: [0xff,0x0f,0x04,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_cmpswap v[2:3], a[2:3] offset:4095

// GFX90A: flat_atomic_add v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x08,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_add v[2:3], a2 offset:4095

// GFX90A: flat_atomic_sub v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_sub v[2:3], a2 offset:4095

// GFX90A: flat_atomic_smin v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x10,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_smin v[2:3], a2 offset:4095

// GFX90A: flat_atomic_umin v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x14,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_umin v[2:3], a2 offset:4095

// GFX90A: flat_atomic_smax v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x18,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_smax v[2:3], a2 offset:4095

// GFX90A: flat_atomic_umax v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_umax v[2:3], a2 offset:4095

// GFX90A: flat_atomic_and v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x20,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_and v[2:3], a2 offset:4095

// GFX90A: flat_atomic_or v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x24,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_or v[2:3], a2 offset:4095

// GFX90A: flat_atomic_xor v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x28,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_xor v[2:3], a2 offset:4095

// GFX90A: flat_atomic_inc v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_inc v[2:3], a2 offset:4095

// GFX90A: flat_atomic_dec v[2:3], a2 offset:4095 ; encoding: [0xff,0x0f,0x30,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_dec v[2:3], a2 offset:4095

// GFX90A: flat_atomic_swap_x2 v[2:3], a[2:3] offset:4095 ; encoding: [0xff,0x0f,0x80,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_swap_x2 v[2:3], a[2:3] offset:4095

// GFX90A: flat_atomic_cmpswap_x2 v[2:3], a[2:5] offset:4095 ; encoding: [0xff,0x0f,0x84,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_cmpswap_x2 v[2:3], a[2:5] offset:4095

// GFX90A: flat_atomic_add_x2 v[2:3], a[2:3] offset:4095 ; encoding: [0xff,0x0f,0x88,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_add_x2 v[2:3], a[2:3] offset:4095

// GFX90A: flat_atomic_sub_x2 v[2:3], a[2:3] offset:4095 ; encoding: [0xff,0x0f,0x8c,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_sub_x2 v[2:3], a[2:3] offset:4095

// GFX90A: flat_atomic_smin_x2 v[2:3], a[2:3] offset:4095 ; encoding: [0xff,0x0f,0x90,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_smin_x2 v[2:3], a[2:3] offset:4095

// GFX90A: flat_atomic_umin_x2 v[2:3], a[2:3] offset:4095 ; encoding: [0xff,0x0f,0x94,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_umin_x2 v[2:3], a[2:3] offset:4095

// GFX90A: flat_atomic_smax_x2 v[2:3], a[2:3] offset:4095 ; encoding: [0xff,0x0f,0x98,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_smax_x2 v[2:3], a[2:3] offset:4095

// GFX90A: flat_atomic_umax_x2 v[2:3], a[2:3] offset:4095 ; encoding: [0xff,0x0f,0x9c,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_umax_x2 v[2:3], a[2:3] offset:4095

// GFX90A: flat_atomic_and_x2 v[2:3], a[2:3] offset:4095 ; encoding: [0xff,0x0f,0xa0,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_and_x2 v[2:3], a[2:3] offset:4095

// GFX90A: flat_atomic_or_x2 v[2:3], a[2:3] offset:4095 ; encoding: [0xff,0x0f,0xa4,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_or_x2 v[2:3], a[2:3] offset:4095

// GFX90A: flat_atomic_xor_x2 v[2:3], a[2:3] offset:4095 ; encoding: [0xff,0x0f,0xa8,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_xor_x2 v[2:3], a[2:3] offset:4095

// GFX90A: flat_atomic_inc_x2 v[2:3], a[2:3] offset:4095 ; encoding: [0xff,0x0f,0xac,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_inc_x2 v[2:3], a[2:3] offset:4095

// GFX90A: flat_atomic_dec_x2 v[2:3], a[2:3] offset:4095 ; encoding: [0xff,0x0f,0xb0,0xdd,0x02,0x02,0x80,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
flat_atomic_dec_x2 v[2:3], a[2:3] offset:4095

// GFX90A: global_load_ubyte a5, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x40,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_ubyte a5, v[2:3], off offset:-1

// GFX90A: global_load_ubyte a255, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x40,0xdc,0x02,0x00,0xff,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_ubyte a255, v[2:3], off offset:-1

// GFX90A: global_load_ubyte a5, v[2:3], off ; encoding: [0x00,0x80,0x40,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_ubyte a5, v[2:3], off

// GFX90A: global_load_sbyte a5, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x44,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_sbyte a5, v[2:3], off offset:-1

// GFX90A: global_load_sbyte a255, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x44,0xdc,0x02,0x00,0xff,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_sbyte a255, v[2:3], off offset:-1

// GFX90A: global_load_sbyte a5, v[2:3], off ; encoding: [0x00,0x80,0x44,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_sbyte a5, v[2:3], off

// GFX90A: global_load_ushort a5, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x48,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_ushort a5, v[2:3], off offset:-1

// GFX90A: global_load_ushort a255, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x48,0xdc,0x02,0x00,0xff,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_ushort a255, v[2:3], off offset:-1

// GFX90A: global_load_ushort a5, v[2:3], off ; encoding: [0x00,0x80,0x48,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_ushort a5, v[2:3], off

// GFX90A: global_load_sshort a5, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x4c,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_sshort a5, v[2:3], off offset:-1

// GFX90A: global_load_sshort a255, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x4c,0xdc,0x02,0x00,0xff,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_sshort a255, v[2:3], off offset:-1

// GFX90A: global_load_sshort a5, v[2:3], off ; encoding: [0x00,0x80,0x4c,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_sshort a5, v[2:3], off

// GFX90A: global_load_dword a5, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x50,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_dword a5, v[2:3], off offset:-1

// GFX90A: global_load_dword a255, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x50,0xdc,0x02,0x00,0xff,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_dword a255, v[2:3], off offset:-1

// GFX90A: global_load_dword a5, v[2:3], off ; encoding: [0x00,0x80,0x50,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_dword a5, v[2:3], off

// GFX90A: global_load_dwordx2 a[6:7], v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x54,0xdc,0x02,0x00,0xff,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_dwordx2 a[6:7], v[2:3], off offset:-1

// GFX90A: global_load_dwordx2 a[254:255], v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x54,0xdc,0x02,0x00,0xff,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_dwordx2 a[254:255], v[2:3], off offset:-1

// GFX90A: global_load_dwordx2 a[6:7], v[2:3], off ; encoding: [0x00,0x80,0x54,0xdc,0x02,0x00,0xff,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_dwordx2 a[6:7], v[2:3], off

// GFX90A: global_load_dwordx3 a[6:8], v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x58,0xdc,0x02,0x00,0xff,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_dwordx3 a[6:8], v[2:3], off offset:-1

// GFX90A: global_load_dwordx3 a[252:254], v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x58,0xdc,0x02,0x00,0xff,0xfc]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_dwordx3 a[252:254], v[2:3], off offset:-1

// GFX90A: global_load_dwordx3 a[6:8], v[2:3], off ; encoding: [0x00,0x80,0x58,0xdc,0x02,0x00,0xff,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_dwordx3 a[6:8], v[2:3], off

// GFX90A: global_load_dwordx4 a[6:9], v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x5c,0xdc,0x02,0x00,0xff,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_dwordx4 a[6:9], v[2:3], off offset:-1

// GFX90A: global_load_dwordx4 a[252:255], v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x5c,0xdc,0x02,0x00,0xff,0xfc]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_dwordx4 a[252:255], v[2:3], off offset:-1

// GFX90A: global_load_dwordx4 a[6:9], v[2:3], off ; encoding: [0x00,0x80,0x5c,0xdc,0x02,0x00,0xff,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_dwordx4 a[6:9], v[2:3], off

// GFX90A: global_store_byte v[2:3], a2, off offset:-1 ; encoding: [0xff,0x9f,0x60,0xdc,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_byte v[2:3], a2, off offset:-1

// GFX90A: global_store_byte v[2:3], a255, off offset:-1 ; encoding: [0xff,0x9f,0x60,0xdc,0x02,0xff,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_byte v[2:3], a255, off offset:-1

// GFX90A: global_store_byte v[2:3], a2, off ; encoding: [0x00,0x80,0x60,0xdc,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_byte v[2:3], a2, off

// GFX90A: global_store_byte_d16_hi v[2:3], a2, off offset:-1 ; encoding: [0xff,0x9f,0x64,0xdc,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_byte_d16_hi v[2:3], a2, off offset:-1

// GFX90A: global_store_byte_d16_hi v[2:3], a255, off offset:-1 ; encoding: [0xff,0x9f,0x64,0xdc,0x02,0xff,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_byte_d16_hi v[2:3], a255, off offset:-1

// GFX90A: global_store_byte_d16_hi v[2:3], a2, off ; encoding: [0x00,0x80,0x64,0xdc,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_byte_d16_hi v[2:3], a2, off

// GFX90A: global_store_short v[2:3], a2, off offset:-1 ; encoding: [0xff,0x9f,0x68,0xdc,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_short v[2:3], a2, off offset:-1

// GFX90A: global_store_short v[2:3], a255, off offset:-1 ; encoding: [0xff,0x9f,0x68,0xdc,0x02,0xff,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_short v[2:3], a255, off offset:-1

// GFX90A: global_store_short v[2:3], a2, off ; encoding: [0x00,0x80,0x68,0xdc,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_short v[2:3], a2, off

// GFX90A: global_store_short_d16_hi v[2:3], a2, off offset:-1 ; encoding: [0xff,0x9f,0x6c,0xdc,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_short_d16_hi v[2:3], a2, off offset:-1

// GFX90A: global_store_short_d16_hi v[2:3], a255, off offset:-1 ; encoding: [0xff,0x9f,0x6c,0xdc,0x02,0xff,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_short_d16_hi v[2:3], a255, off offset:-1

// GFX90A: global_store_short_d16_hi v[2:3], a2, off ; encoding: [0x00,0x80,0x6c,0xdc,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_short_d16_hi v[2:3], a2, off

// GFX90A: global_store_dword v[2:3], a2, off offset:-1 ; encoding: [0xff,0x9f,0x70,0xdc,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_dword v[2:3], a2, off offset:-1

// GFX90A: global_store_dword v[2:3], a255, off offset:-1 ; encoding: [0xff,0x9f,0x70,0xdc,0x02,0xff,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_dword v[2:3], a255, off offset:-1

// GFX90A: global_store_dword v[2:3], a2, off ; encoding: [0x00,0x80,0x70,0xdc,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_dword v[2:3], a2, off

// GFX90A: global_store_dwordx2 v[2:3], a[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x74,0xdc,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_dwordx2 v[2:3], a[2:3], off offset:-1

// GFX90A: global_store_dwordx2 v[2:3], a[254:255], off offset:-1 ; encoding: [0xff,0x9f,0x74,0xdc,0x02,0xfe,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_dwordx2 v[2:3], a[254:255], off offset:-1

// GFX90A: global_store_dwordx2 v[2:3], a[2:3], off ; encoding: [0x00,0x80,0x74,0xdc,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_dwordx2 v[2:3], a[2:3], off

// GFX90A: global_store_dwordx3 v[2:3], a[2:4], off offset:-1 ; encoding: [0xff,0x9f,0x78,0xdc,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_dwordx3 v[2:3], a[2:4], off offset:-1

// GFX90A: global_store_dwordx3 v[2:3], a[252:254], off offset:-1 ; encoding: [0xff,0x9f,0x78,0xdc,0x02,0xfc,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_dwordx3 v[2:3], a[252:254], off offset:-1

// GFX90A: global_store_dwordx3 v[2:3], a[2:4], off ; encoding: [0x00,0x80,0x78,0xdc,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_dwordx3 v[2:3], a[2:4], off

// GFX90A: global_store_dwordx4 v[2:3], a[2:5], off offset:-1 ; encoding: [0xff,0x9f,0x7c,0xdc,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_dwordx4 v[2:3], a[2:5], off offset:-1

// GFX90A: global_store_dwordx4 v[2:3], a[252:255], off offset:-1 ; encoding: [0xff,0x9f,0x7c,0xdc,0x02,0xfc,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_dwordx4 v[2:3], a[252:255], off offset:-1

// GFX90A: global_store_dwordx4 v[2:3], a[2:5], off ; encoding: [0x00,0x80,0x7c,0xdc,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_store_dwordx4 v[2:3], a[2:5], off

// GFX90A: global_load_ubyte_d16 a5, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x80,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_ubyte_d16 a5, v[2:3], off offset:-1

// GFX90A: global_load_ubyte_d16 a255, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x80,0xdc,0x02,0x00,0xff,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_ubyte_d16 a255, v[2:3], off offset:-1

// GFX90A: global_load_ubyte_d16 a5, v[2:3], off ; encoding: [0x00,0x80,0x80,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_ubyte_d16 a5, v[2:3], off

// GFX90A: global_load_ubyte_d16_hi a5, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x84,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_ubyte_d16_hi a5, v[2:3], off offset:-1

// GFX90A: global_load_ubyte_d16_hi a255, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x84,0xdc,0x02,0x00,0xff,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_ubyte_d16_hi a255, v[2:3], off offset:-1

// GFX90A: global_load_ubyte_d16_hi a5, v[2:3], off ; encoding: [0x00,0x80,0x84,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_ubyte_d16_hi a5, v[2:3], off

// GFX90A: global_load_sbyte_d16 a5, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x88,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_sbyte_d16 a5, v[2:3], off offset:-1

// GFX90A: global_load_sbyte_d16 a255, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x88,0xdc,0x02,0x00,0xff,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_sbyte_d16 a255, v[2:3], off offset:-1

// GFX90A: global_load_sbyte_d16 a5, v[2:3], off ; encoding: [0x00,0x80,0x88,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_sbyte_d16 a5, v[2:3], off

// GFX90A: global_load_sbyte_d16_hi a5, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x8c,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_sbyte_d16_hi a5, v[2:3], off offset:-1

// GFX90A: global_load_sbyte_d16_hi a255, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x8c,0xdc,0x02,0x00,0xff,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_sbyte_d16_hi a255, v[2:3], off offset:-1

// GFX90A: global_load_sbyte_d16_hi a5, v[2:3], off ; encoding: [0x00,0x80,0x8c,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_sbyte_d16_hi a5, v[2:3], off

// GFX90A: global_load_short_d16 a5, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x90,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_short_d16 a5, v[2:3], off offset:-1

// GFX90A: global_load_short_d16 a255, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x90,0xdc,0x02,0x00,0xff,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_short_d16 a255, v[2:3], off offset:-1

// GFX90A: global_load_short_d16 a5, v[2:3], off ; encoding: [0x00,0x80,0x90,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_short_d16 a5, v[2:3], off

// GFX90A: global_load_short_d16_hi a5, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x94,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_short_d16_hi a5, v[2:3], off offset:-1

// GFX90A: global_load_short_d16_hi a255, v[2:3], off offset:-1 ; encoding: [0xff,0x9f,0x94,0xdc,0x02,0x00,0xff,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_short_d16_hi a255, v[2:3], off offset:-1

// GFX90A: global_load_short_d16_hi a5, v[2:3], off ; encoding: [0x00,0x80,0x94,0xdc,0x02,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_load_short_d16_hi a5, v[2:3], off

// GFX90A: global_atomic_swap a1, v[2:3], a2, off glc ; encoding: [0x00,0x80,0x01,0xdd,0x02,0x02,0xff,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_swap a1, v[2:3], a2, off glc

// GFX90A: global_atomic_cmpswap a1, v[2:3], a[2:3], off glc ; encoding: [0x00,0x80,0x05,0xdd,0x02,0x02,0xff,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_cmpswap a1, v[2:3], a[2:3], off glc

// GFX90A: global_atomic_add a1, v[2:3], a2, off glc ; encoding: [0x00,0x80,0x09,0xdd,0x02,0x02,0xff,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_add a1, v[2:3], a2, off glc

// GFX90A: global_atomic_sub a1, v[2:3], a2, off glc ; encoding: [0x00,0x80,0x0d,0xdd,0x02,0x02,0xff,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_sub a1, v[2:3], a2, off glc

// GFX90A: global_atomic_smin a1, v[2:3], a2, off glc ; encoding: [0x00,0x80,0x11,0xdd,0x02,0x02,0xff,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_smin a1, v[2:3], a2, off glc

// GFX90A: global_atomic_umin a1, v[2:3], a2, off glc ; encoding: [0x00,0x80,0x15,0xdd,0x02,0x02,0xff,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_umin a1, v[2:3], a2, off glc

// GFX90A: global_atomic_smax a1, v[2:3], a2, off glc ; encoding: [0x00,0x80,0x19,0xdd,0x02,0x02,0xff,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_smax a1, v[2:3], a2, off glc

// GFX90A: global_atomic_umax a1, v[2:3], a2, off glc ; encoding: [0x00,0x80,0x1d,0xdd,0x02,0x02,0xff,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_umax a1, v[2:3], a2, off glc

// GFX90A: global_atomic_and a1, v[2:3], a2, off glc ; encoding: [0x00,0x80,0x21,0xdd,0x02,0x02,0xff,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_and a1, v[2:3], a2, off glc

// GFX90A: global_atomic_or a1, v[2:3], a2, off glc ; encoding: [0x00,0x80,0x25,0xdd,0x02,0x02,0xff,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_or a1, v[2:3], a2, off glc

// GFX90A: global_atomic_xor a1, v[2:3], a2, off glc ; encoding: [0x00,0x80,0x29,0xdd,0x02,0x02,0xff,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_xor a1, v[2:3], a2, off glc

// GFX90A: global_atomic_inc a1, v[2:3], a2, off glc ; encoding: [0x00,0x80,0x2d,0xdd,0x02,0x02,0xff,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_inc a1, v[2:3], a2, off glc

// GFX90A: global_atomic_dec a1, v[2:3], a2, off glc ; encoding: [0x00,0x80,0x31,0xdd,0x02,0x02,0xff,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_dec a1, v[2:3], a2, off glc

// GFX90A: global_atomic_swap_x2 a[2:3], v[2:3], a[2:3], off glc ; encoding: [0x00,0x80,0x81,0xdd,0x02,0x02,0xff,0x02]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_swap_x2 a[2:3], v[2:3], a[2:3], off glc

// GFX90A: global_atomic_cmpswap_x2 a[2:3], v[2:3], a[2:5], off glc ; encoding: [0x00,0x80,0x85,0xdd,0x02,0x02,0xff,0x02]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_cmpswap_x2 a[2:3], v[2:3], a[2:5], off glc

// GFX90A: global_atomic_add_x2 a[2:3], v[2:3], a[2:3], off glc ; encoding: [0x00,0x80,0x89,0xdd,0x02,0x02,0xff,0x02]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_add_x2 a[2:3], v[2:3], a[2:3], off glc

// GFX90A: global_atomic_sub_x2 a[2:3], v[2:3], a[2:3], off glc ; encoding: [0x00,0x80,0x8d,0xdd,0x02,0x02,0xff,0x02]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_sub_x2 a[2:3], v[2:3], a[2:3], off glc

// GFX90A: global_atomic_smin_x2 a[2:3], v[2:3], a[2:3], off glc ; encoding: [0x00,0x80,0x91,0xdd,0x02,0x02,0xff,0x02]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_smin_x2 a[2:3], v[2:3], a[2:3], off glc

// GFX90A: global_atomic_umin_x2 a[2:3], v[2:3], a[2:3], off glc ; encoding: [0x00,0x80,0x95,0xdd,0x02,0x02,0xff,0x02]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_umin_x2 a[2:3], v[2:3], a[2:3], off glc

// GFX90A: global_atomic_smax_x2 a[2:3], v[2:3], a[2:3], off glc ; encoding: [0x00,0x80,0x99,0xdd,0x02,0x02,0xff,0x02]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_smax_x2 a[2:3], v[2:3], a[2:3], off glc

// GFX90A: global_atomic_umax_x2 a[2:3], v[2:3], a[2:3], off glc ; encoding: [0x00,0x80,0x9d,0xdd,0x02,0x02,0xff,0x02]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_umax_x2 a[2:3], v[2:3], a[2:3], off glc

// GFX90A: global_atomic_and_x2 a[2:3], v[2:3], a[2:3], off glc ; encoding: [0x00,0x80,0xa1,0xdd,0x02,0x02,0xff,0x02]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_and_x2 a[2:3], v[2:3], a[2:3], off glc

// GFX90A: global_atomic_or_x2 a[2:3], v[2:3], a[2:3], off glc ; encoding: [0x00,0x80,0xa5,0xdd,0x02,0x02,0xff,0x02]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_or_x2 a[2:3], v[2:3], a[2:3], off glc

// GFX90A: global_atomic_xor_x2 a[2:3], v[2:3], a[2:3], off glc ; encoding: [0x00,0x80,0xa9,0xdd,0x02,0x02,0xff,0x02]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_xor_x2 a[2:3], v[2:3], a[2:3], off glc

// GFX90A: global_atomic_inc_x2 a[2:3], v[2:3], a[2:3], off glc ; encoding: [0x00,0x80,0xad,0xdd,0x02,0x02,0xff,0x02]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_inc_x2 a[2:3], v[2:3], a[2:3], off glc

// GFX90A: global_atomic_dec_x2 a[2:3], v[2:3], a[2:3], off glc ; encoding: [0x00,0x80,0xb1,0xdd,0x02,0x02,0xff,0x02]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_dec_x2 a[2:3], v[2:3], a[2:3], off glc

// GFX90A: global_atomic_swap v[2:3], a2, off ; encoding: [0x00,0x80,0x00,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_swap v[2:3], a2, off

// GFX90A: global_atomic_cmpswap v[2:3], a[2:3], off ; encoding: [0x00,0x80,0x04,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_cmpswap v[2:3], a[2:3], off

// GFX90A: global_atomic_add v[2:3], a2, off ; encoding: [0x00,0x80,0x08,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_add v[2:3], a2, off

// GFX90A: global_atomic_sub v[2:3], a2, off ; encoding: [0x00,0x80,0x0c,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_sub v[2:3], a2, off

// GFX90A: global_atomic_smin v[2:3], a2, off ; encoding: [0x00,0x80,0x10,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_smin v[2:3], a2, off

// GFX90A: global_atomic_umin v[2:3], a2, off ; encoding: [0x00,0x80,0x14,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_umin v[2:3], a2, off

// GFX90A: global_atomic_smax v[2:3], a2, off ; encoding: [0x00,0x80,0x18,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_smax v[2:3], a2, off

// GFX90A: global_atomic_umax v[2:3], a2, off ; encoding: [0x00,0x80,0x1c,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_umax v[2:3], a2, off

// GFX90A: global_atomic_and v[2:3], a2, off ; encoding: [0x00,0x80,0x20,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_and v[2:3], a2, off

// GFX90A: global_atomic_or v[2:3], a2, off ; encoding: [0x00,0x80,0x24,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_or v[2:3], a2, off

// GFX90A: global_atomic_xor v[2:3], a2, off ; encoding: [0x00,0x80,0x28,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_xor v[2:3], a2, off

// GFX90A: global_atomic_inc v[2:3], a2, off ; encoding: [0x00,0x80,0x2c,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_inc v[2:3], a2, off

// GFX90A: global_atomic_dec v[2:3], a2, off ; encoding: [0x00,0x80,0x30,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_dec v[2:3], a2, off

// GFX90A: global_atomic_swap_x2 v[2:3], a[2:3], off ; encoding: [0x00,0x80,0x80,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_swap_x2 v[2:3], a[2:3], off

// GFX90A: global_atomic_cmpswap_x2 v[2:3], a[2:5], off ; encoding: [0x00,0x80,0x84,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_cmpswap_x2 v[2:3], a[2:5], off

// GFX90A: global_atomic_add_x2 v[2:3], a[2:3], off ; encoding: [0x00,0x80,0x88,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_add_x2 v[2:3], a[2:3], off

// GFX90A: global_atomic_sub_x2 v[2:3], a[2:3], off ; encoding: [0x00,0x80,0x8c,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_sub_x2 v[2:3], a[2:3], off

// GFX90A: global_atomic_smin_x2 v[2:3], a[2:3], off ; encoding: [0x00,0x80,0x90,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_smin_x2 v[2:3], a[2:3], off

// GFX90A: global_atomic_umin_x2 v[2:3], a[2:3], off ; encoding: [0x00,0x80,0x94,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_umin_x2 v[2:3], a[2:3], off

// GFX90A: global_atomic_smax_x2 v[2:3], a[2:3], off ; encoding: [0x00,0x80,0x98,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_smax_x2 v[2:3], a[2:3], off

// GFX90A: global_atomic_umax_x2 v[2:3], a[2:3], off ; encoding: [0x00,0x80,0x9c,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_umax_x2 v[2:3], a[2:3], off

// GFX90A: global_atomic_and_x2 v[2:3], a[2:3], off ; encoding: [0x00,0x80,0xa0,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_and_x2 v[2:3], a[2:3], off

// GFX90A: global_atomic_or_x2 v[2:3], a[2:3], off ; encoding: [0x00,0x80,0xa4,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_or_x2 v[2:3], a[2:3], off

// GFX90A: global_atomic_xor_x2 v[2:3], a[2:3], off ; encoding: [0x00,0x80,0xa8,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_xor_x2 v[2:3], a[2:3], off

// GFX90A: global_atomic_inc_x2 v[2:3], a[2:3], off ; encoding: [0x00,0x80,0xac,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_inc_x2 v[2:3], a[2:3], off

// GFX90A: global_atomic_dec_x2 v[2:3], a[2:3], off ; encoding: [0x00,0x80,0xb0,0xdd,0x02,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
global_atomic_dec_x2 v[2:3], a[2:3], off

// GFX90A: scratch_load_ubyte a5, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x40,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte a5, off, s2 offset:-1

// GFX90A: scratch_load_ubyte a255, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x40,0xdc,0x00,0x00,0x82,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte a255, off, s2 offset:-1

// GFX90A: scratch_load_ubyte a5, off, s101 offset:-1 ; encoding: [0xff,0x5f,0x40,0xdc,0x00,0x00,0xe5,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte a5, off, s101 offset:-1

// GFX90A: scratch_load_ubyte a5, off, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x40,0xdc,0x00,0x00,0xe6,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte a5, off, flat_scratch_lo offset:-1

// GFX90A: scratch_load_ubyte a5, off, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x40,0xdc,0x00,0x00,0xe7,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte a5, off, flat_scratch_hi offset:-1

// GFX90A: scratch_load_ubyte a5, off, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x40,0xdc,0x00,0x00,0xea,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte a5, off, vcc_lo offset:-1

// GFX90A: scratch_load_ubyte a5, off, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x40,0xdc,0x00,0x00,0xeb,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte a5, off, vcc_hi offset:-1

// GFX90A: scratch_load_ubyte a5, v0, off offset:-1 ; encoding: [0xff,0x5f,0x40,0xdc,0x00,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte a5, v0, off offset:-1

// GFX90A: scratch_load_ubyte a5, off, s2  ; encoding: [0x00,0x40,0x40,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte a5, off, s2

// GFX90A: scratch_load_ubyte a5, off, s2  ; encoding: [0x00,0x40,0x40,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte a5, off, s2

// GFX90A: scratch_load_ubyte a5, off, s2 offset:4095 ; encoding: [0xff,0x4f,0x40,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte a5, off, s2 offset:4095

// GFX90A: scratch_load_ubyte a5, off, s2 offset:-4096 ; encoding: [0x00,0x50,0x40,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte a5, off, s2 offset:-4096

// GFX90A: scratch_load_ubyte a5, off, s2 offset:-1 glc ; encoding: [0xff,0x5f,0x41,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte a5, off, s2 offset:-1 glc

// GFX90A: scratch_load_ubyte a5, off, s2 offset:-1 slc ; encoding: [0xff,0x5f,0x42,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte a5, off, s2 offset:-1 slc

// GFX90A: scratch_load_sbyte a5, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x44,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte a5, off, s2 offset:-1

// GFX90A: scratch_load_sbyte a255, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x44,0xdc,0x00,0x00,0x82,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte a255, off, s2 offset:-1

// GFX90A: scratch_load_sbyte a5, off, s101 offset:-1 ; encoding: [0xff,0x5f,0x44,0xdc,0x00,0x00,0xe5,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte a5, off, s101 offset:-1

// GFX90A: scratch_load_sbyte a5, off, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x44,0xdc,0x00,0x00,0xe6,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte a5, off, flat_scratch_lo offset:-1

// GFX90A: scratch_load_sbyte a5, off, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x44,0xdc,0x00,0x00,0xe7,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte a5, off, flat_scratch_hi offset:-1

// GFX90A: scratch_load_sbyte a5, off, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x44,0xdc,0x00,0x00,0xea,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte a5, off, vcc_lo offset:-1

// GFX90A: scratch_load_sbyte a5, off, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x44,0xdc,0x00,0x00,0xeb,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte a5, off, vcc_hi offset:-1

// GFX90A: scratch_load_sbyte a5, v0, off offset:-1 ; encoding: [0xff,0x5f,0x44,0xdc,0x00,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte a5, v0, off offset:-1

// GFX90A: scratch_load_sbyte a5, off, s2  ; encoding: [0x00,0x40,0x44,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte a5, off, s2

// GFX90A: scratch_load_sbyte a5, off, s2  ; encoding: [0x00,0x40,0x44,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte a5, off, s2

// GFX90A: scratch_load_sbyte a5, off, s2 offset:4095 ; encoding: [0xff,0x4f,0x44,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte a5, off, s2 offset:4095

// GFX90A: scratch_load_sbyte a5, off, s2 offset:-4096 ; encoding: [0x00,0x50,0x44,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte a5, off, s2 offset:-4096

// GFX90A: scratch_load_sbyte a5, off, s2 offset:-1 glc ; encoding: [0xff,0x5f,0x45,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte a5, off, s2 offset:-1 glc

// GFX90A: scratch_load_sbyte a5, off, s2 offset:-1 slc ; encoding: [0xff,0x5f,0x46,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte a5, off, s2 offset:-1 slc

// GFX90A: scratch_load_ushort a5, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x48,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ushort a5, off, s2 offset:-1

// GFX90A: scratch_load_ushort a255, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x48,0xdc,0x00,0x00,0x82,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ushort a255, off, s2 offset:-1

// GFX90A: scratch_load_ushort a5, off, s101 offset:-1 ; encoding: [0xff,0x5f,0x48,0xdc,0x00,0x00,0xe5,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ushort a5, off, s101 offset:-1

// GFX90A: scratch_load_ushort a5, off, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x48,0xdc,0x00,0x00,0xe6,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ushort a5, off, flat_scratch_lo offset:-1

// GFX90A: scratch_load_ushort a5, off, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x48,0xdc,0x00,0x00,0xe7,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ushort a5, off, flat_scratch_hi offset:-1

// GFX90A: scratch_load_ushort a5, off, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x48,0xdc,0x00,0x00,0xea,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ushort a5, off, vcc_lo offset:-1

// GFX90A: scratch_load_ushort a5, off, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x48,0xdc,0x00,0x00,0xeb,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ushort a5, off, vcc_hi offset:-1

// GFX90A: scratch_load_ushort a5, v0, off offset:-1 ; encoding: [0xff,0x5f,0x48,0xdc,0x00,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ushort a5, v0, off offset:-1

// GFX90A: scratch_load_ushort a5, off, s2 ; encoding: [0x00,0x40,0x48,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ushort a5, off, s2

// GFX90A: scratch_load_ushort a5, off, s2 ; encoding: [0x00,0x40,0x48,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ushort a5, off, s2

// GFX90A: scratch_load_ushort a5, off, s2 offset:4095 ; encoding: [0xff,0x4f,0x48,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ushort a5, off, s2 offset:4095

// GFX90A: scratch_load_ushort a5, off, s2 offset:-4096 ; encoding: [0x00,0x50,0x48,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ushort a5, off, s2 offset:-4096

// GFX90A: scratch_load_ushort a5, off, s2 offset:-1 glc ; encoding: [0xff,0x5f,0x49,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ushort a5, off, s2 offset:-1 glc

// GFX90A: scratch_load_ushort a5, off, s2 offset:-1 slc ; encoding: [0xff,0x5f,0x4a,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ushort a5, off, s2 offset:-1 slc

// GFX90A: scratch_load_sshort a5, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x4c,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sshort a5, off, s2 offset:-1

// GFX90A: scratch_load_sshort a255, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x4c,0xdc,0x00,0x00,0x82,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sshort a255, off, s2 offset:-1

// GFX90A: scratch_load_sshort a5, off, s101 offset:-1 ; encoding: [0xff,0x5f,0x4c,0xdc,0x00,0x00,0xe5,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sshort a5, off, s101 offset:-1

// GFX90A: scratch_load_sshort a5, off, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x4c,0xdc,0x00,0x00,0xe6,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sshort a5, off, flat_scratch_lo offset:-1

// GFX90A: scratch_load_sshort a5, off, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x4c,0xdc,0x00,0x00,0xe7,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sshort a5, off, flat_scratch_hi offset:-1

// GFX90A: scratch_load_sshort a5, off, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x4c,0xdc,0x00,0x00,0xea,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sshort a5, off, vcc_lo offset:-1

// GFX90A: scratch_load_sshort a5, off, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x4c,0xdc,0x00,0x00,0xeb,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sshort a5, off, vcc_hi offset:-1

// GFX90A: scratch_load_sshort a5, v0, off offset:-1 ; encoding: [0xff,0x5f,0x4c,0xdc,0x00,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sshort a5, v0, off offset:-1

// GFX90A: scratch_load_sshort a5, off, s2 ; encoding: [0x00,0x40,0x4c,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sshort a5, off, s2

// GFX90A: scratch_load_sshort a5, off, s2 ; encoding: [0x00,0x40,0x4c,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sshort a5, off, s2

// GFX90A: scratch_load_sshort a5, off, s2 offset:4095 ; encoding: [0xff,0x4f,0x4c,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sshort a5, off, s2 offset:4095

// GFX90A: scratch_load_sshort a5, off, s2 offset:-4096 ; encoding: [0x00,0x50,0x4c,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sshort a5, off, s2 offset:-4096

// GFX90A: scratch_load_sshort a5, off, s2 offset:-1 glc ; encoding: [0xff,0x5f,0x4d,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sshort a5, off, s2 offset:-1 glc

// GFX90A: scratch_load_sshort a5, off, s2 offset:-1 slc ; encoding: [0xff,0x5f,0x4e,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sshort a5, off, s2 offset:-1 slc

// GFX90A: scratch_load_dword a5, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x50,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dword a5, off, s2 offset:-1

// GFX90A: scratch_load_dword a255, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x50,0xdc,0x00,0x00,0x82,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dword a255, off, s2 offset:-1

// GFX90A: scratch_load_dword a5, off, s101 offset:-1 ; encoding: [0xff,0x5f,0x50,0xdc,0x00,0x00,0xe5,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dword a5, off, s101 offset:-1

// GFX90A: scratch_load_dword a5, off, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x50,0xdc,0x00,0x00,0xe6,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dword a5, off, flat_scratch_lo offset:-1

// GFX90A: scratch_load_dword a5, off, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x50,0xdc,0x00,0x00,0xe7,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dword a5, off, flat_scratch_hi offset:-1

// GFX90A: scratch_load_dword a5, off, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x50,0xdc,0x00,0x00,0xea,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dword a5, off, vcc_lo offset:-1

// GFX90A: scratch_load_dword a5, off, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x50,0xdc,0x00,0x00,0xeb,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dword a5, off, vcc_hi offset:-1

// GFX90A: scratch_load_dword a5, v0, off offset:-1 ; encoding: [0xff,0x5f,0x50,0xdc,0x00,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dword a5, v0, off offset:-1

// GFX90A: scratch_load_dword a5, off, s2  ; encoding: [0x00,0x40,0x50,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dword a5, off, s2

// GFX90A: scratch_load_dword a5, off, s2  ; encoding: [0x00,0x40,0x50,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dword a5, off, s2

// GFX90A: scratch_load_dword a5, off, s2 offset:4095 ; encoding: [0xff,0x4f,0x50,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dword a5, off, s2 offset:4095

// GFX90A: scratch_load_dword a5, off, s2 offset:-4096 ; encoding: [0x00,0x50,0x50,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dword a5, off, s2 offset:-4096

// GFX90A: scratch_load_dword a5, off, s2 offset:-1 glc ; encoding: [0xff,0x5f,0x51,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dword a5, off, s2 offset:-1 glc

// GFX90A: scratch_load_dword a5, off, s2 offset:-1 slc ; encoding: [0xff,0x5f,0x52,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dword a5, off, s2 offset:-1 slc

// GFX90A: scratch_load_dwordx2 a[6:7], off, s2 offset:-1 ; encoding: [0xff,0x5f,0x54,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx2 a[6:7], off, s2 offset:-1

// GFX90A: scratch_load_dwordx2 a[254:255], off, s2 offset:-1 ; encoding: [0xff,0x5f,0x54,0xdc,0x00,0x00,0x82,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx2 a[254:255], off, s2 offset:-1

// GFX90A: scratch_load_dwordx2 a[6:7], off, s101 offset:-1 ; encoding: [0xff,0x5f,0x54,0xdc,0x00,0x00,0xe5,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx2 a[6:7], off, s101 offset:-1

// GFX90A: scratch_load_dwordx2 a[6:7], off, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x54,0xdc,0x00,0x00,0xe6,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx2 a[6:7], off, flat_scratch_lo offset:-1

// GFX90A: scratch_load_dwordx2 a[6:7], off, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x54,0xdc,0x00,0x00,0xe7,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx2 a[6:7], off, flat_scratch_hi offset:-1

// GFX90A: scratch_load_dwordx2 a[6:7], off, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x54,0xdc,0x00,0x00,0xea,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx2 a[6:7], off, vcc_lo offset:-1

// GFX90A: scratch_load_dwordx2 a[6:7], off, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x54,0xdc,0x00,0x00,0xeb,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx2 a[6:7], off, vcc_hi offset:-1

// GFX90A: scratch_load_dwordx2 a[6:7], v0, off offset:-1 ; encoding: [0xff,0x5f,0x54,0xdc,0x00,0x00,0xff,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx2 a[6:7], v0, off offset:-1

// GFX90A: scratch_load_dwordx2 a[6:7], off, s2 ; encoding: [0x00,0x40,0x54,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx2 a[6:7], off, s2

// GFX90A: scratch_load_dwordx2 a[6:7], off, s2 ; encoding: [0x00,0x40,0x54,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx2 a[6:7], off, s2

// GFX90A: scratch_load_dwordx2 a[6:7], off, s2 offset:4095 ; encoding: [0xff,0x4f,0x54,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx2 a[6:7], off, s2 offset:4095

// GFX90A: scratch_load_dwordx2 a[6:7], off, s2 offset:-4096 ; encoding: [0x00,0x50,0x54,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx2 a[6:7], off, s2 offset:-4096

// GFX90A: scratch_load_dwordx2 a[6:7], off, s2 offset:-1 glc ; encoding: [0xff,0x5f,0x55,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx2 a[6:7], off, s2 offset:-1 glc

// GFX90A: scratch_load_dwordx2 a[6:7], off, s2 offset:-1 slc ; encoding: [0xff,0x5f,0x56,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx2 a[6:7], off, s2 offset:-1 slc

// GFX90A: scratch_load_dwordx3 a[6:8], off, s2 offset:-1 ; encoding: [0xff,0x5f,0x58,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx3 a[6:8], off, s2 offset:-1

// GFX90A: scratch_load_dwordx3 a[252:254], off, s2 offset:-1 ; encoding: [0xff,0x5f,0x58,0xdc,0x00,0x00,0x82,0xfc]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx3 a[252:254], off, s2 offset:-1

// GFX90A: scratch_load_dwordx3 a[6:8], off, s101 offset:-1 ; encoding: [0xff,0x5f,0x58,0xdc,0x00,0x00,0xe5,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx3 a[6:8], off, s101 offset:-1

// GFX90A: scratch_load_dwordx3 a[6:8], off, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x58,0xdc,0x00,0x00,0xe6,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx3 a[6:8], off, flat_scratch_lo offset:-1

// GFX90A: scratch_load_dwordx3 a[6:8], off, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x58,0xdc,0x00,0x00,0xe7,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx3 a[6:8], off, flat_scratch_hi offset:-1

// GFX90A: scratch_load_dwordx3 a[6:8], off, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x58,0xdc,0x00,0x00,0xea,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx3 a[6:8], off, vcc_lo offset:-1

// GFX90A: scratch_load_dwordx3 a[6:8], off, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x58,0xdc,0x00,0x00,0xeb,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx3 a[6:8], off, vcc_hi offset:-1

// GFX90A: scratch_load_dwordx3 a[6:8], v0, off offset:-1 ; encoding: [0xff,0x5f,0x58,0xdc,0x00,0x00,0xff,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx3 a[6:8], v0, off offset:-1

// GFX90A: scratch_load_dwordx3 a[6:8], off, s2 ; encoding: [0x00,0x40,0x58,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx3 a[6:8], off, s2

// GFX90A: scratch_load_dwordx3 a[6:8], off, s2 ; encoding: [0x00,0x40,0x58,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx3 a[6:8], off, s2

// GFX90A: scratch_load_dwordx3 a[6:8], off, s2 offset:4095 ; encoding: [0xff,0x4f,0x58,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx3 a[6:8], off, s2 offset:4095

// GFX90A: scratch_load_dwordx3 a[6:8], off, s2 offset:-4096 ; encoding: [0x00,0x50,0x58,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx3 a[6:8], off, s2 offset:-4096

// GFX90A: scratch_load_dwordx3 a[6:8], off, s2 offset:-1 glc ; encoding: [0xff,0x5f,0x59,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx3 a[6:8], off, s2 offset:-1 glc

// GFX90A: scratch_load_dwordx3 a[6:8], off, s2 offset:-1 slc ; encoding: [0xff,0x5f,0x5a,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx3 a[6:8], off, s2 offset:-1 slc

// GFX90A: scratch_load_dwordx4 a[6:9], off, s2 offset:-1 ; encoding: [0xff,0x5f,0x5c,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx4 a[6:9], off, s2 offset:-1

// GFX90A: scratch_load_dwordx4 a[252:255], off, s2 offset:-1 ; encoding: [0xff,0x5f,0x5c,0xdc,0x00,0x00,0x82,0xfc]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx4 a[252:255], off, s2 offset:-1

// GFX90A: scratch_load_dwordx4 a[6:9], off, s101 offset:-1 ; encoding: [0xff,0x5f,0x5c,0xdc,0x00,0x00,0xe5,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx4 a[6:9], off, s101 offset:-1

// GFX90A: scratch_load_dwordx4 a[6:9], off, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x5c,0xdc,0x00,0x00,0xe6,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx4 a[6:9], off, flat_scratch_lo offset:-1

// GFX90A: scratch_load_dwordx4 a[6:9], off, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x5c,0xdc,0x00,0x00,0xe7,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx4 a[6:9], off, flat_scratch_hi offset:-1

// GFX90A: scratch_load_dwordx4 a[6:9], off, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x5c,0xdc,0x00,0x00,0xea,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx4 a[6:9], off, vcc_lo offset:-1

// GFX90A: scratch_load_dwordx4 a[6:9], off, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x5c,0xdc,0x00,0x00,0xeb,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx4 a[6:9], off, vcc_hi offset:-1

// GFX90A: scratch_load_dwordx4 a[6:9], v0, off offset:-1 ; encoding: [0xff,0x5f,0x5c,0xdc,0x00,0x00,0xff,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx4 a[6:9], v0, off offset:-1

// GFX90A: scratch_load_dwordx4 a[6:9], off, s2 ; encoding: [0x00,0x40,0x5c,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx4 a[6:9], off, s2

// GFX90A: scratch_load_dwordx4 a[6:9], off, s2 ; encoding: [0x00,0x40,0x5c,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx4 a[6:9], off, s2

// GFX90A: scratch_load_dwordx4 a[6:9], off, s2 offset:4095 ; encoding: [0xff,0x4f,0x5c,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx4 a[6:9], off, s2 offset:4095

// GFX90A: scratch_load_dwordx4 a[6:9], off, s2 offset:-4096 ; encoding: [0x00,0x50,0x5c,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx4 a[6:9], off, s2 offset:-4096

// GFX90A: scratch_load_dwordx4 a[6:9], off, s2 offset:-1 glc ; encoding: [0xff,0x5f,0x5d,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx4 a[6:9], off, s2 offset:-1 glc

// GFX90A: scratch_load_dwordx4 a[6:9], off, s2 offset:-1 slc ; encoding: [0xff,0x5f,0x5e,0xdc,0x00,0x00,0x82,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_dwordx4 a[6:9], off, s2 offset:-1 slc

// GFX90A: scratch_store_byte off, a2, s3 offset:-1 ; encoding: [0xff,0x5f,0x60,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte off, a2, s3 offset:-1

// GFX90A: scratch_store_byte off, a255, s3 offset:-1 ; encoding: [0xff,0x5f,0x60,0xdc,0x00,0xff,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte off, a255, s3 offset:-1

// GFX90A: scratch_store_byte off, a2, s101 offset:-1 ; encoding: [0xff,0x5f,0x60,0xdc,0x00,0x02,0xe5,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte off, a2, s101 offset:-1

// GFX90A: scratch_store_byte off, a2, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x60,0xdc,0x00,0x02,0xe6,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte off, a2, flat_scratch_lo offset:-1

// GFX90A: scratch_store_byte off, a2, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x60,0xdc,0x00,0x02,0xe7,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte off, a2, flat_scratch_hi offset:-1

// GFX90A: scratch_store_byte off, a2, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x60,0xdc,0x00,0x02,0xea,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte off, a2, vcc_lo offset:-1

// GFX90A: scratch_store_byte off, a2, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x60,0xdc,0x00,0x02,0xeb,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte off, a2, vcc_hi offset:-1

// GFX90A: scratch_store_byte v0, a2, off offset:-1 ; encoding: [0xff,0x5f,0x60,0xdc,0x00,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte v0, a2, off offset:-1

// GFX90A: scratch_store_byte off, a2, s3  ; encoding: [0x00,0x40,0x60,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte off, a2, s3

// GFX90A: scratch_store_byte off, a2, s3  ; encoding: [0x00,0x40,0x60,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte off, a2, s3

// GFX90A: scratch_store_byte off, a2, s3 offset:4095 ; encoding: [0xff,0x4f,0x60,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte off, a2, s3 offset:4095

// GFX90A: scratch_store_byte off, a2, s3 offset:-4096 ; encoding: [0x00,0x50,0x60,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte off, a2, s3 offset:-4096

// GFX90A: scratch_store_byte off, a2, s3 offset:-1 glc ; encoding: [0xff,0x5f,0x61,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte off, a2, s3 offset:-1 glc

// GFX90A: scratch_store_byte off, a2, s3 offset:-1 slc ; encoding: [0xff,0x5f,0x62,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte off, a2, s3 offset:-1 slc

// GFX90A: scratch_store_byte_d16_hi off, a2, s3 offset:-1 ; encoding: [0xff,0x5f,0x64,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte_d16_hi off, a2, s3 offset:-1

// GFX90A: scratch_store_byte_d16_hi off, a255, s3 offset:-1 ; encoding: [0xff,0x5f,0x64,0xdc,0x00,0xff,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte_d16_hi off, a255, s3 offset:-1

// GFX90A: scratch_store_byte_d16_hi off, a2, s101 offset:-1 ; encoding: [0xff,0x5f,0x64,0xdc,0x00,0x02,0xe5,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte_d16_hi off, a2, s101 offset:-1

// GFX90A: scratch_store_byte_d16_hi off, a2, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x64,0xdc,0x00,0x02,0xe6,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte_d16_hi off, a2, flat_scratch_lo offset:-1

// GFX90A: scratch_store_byte_d16_hi off, a2, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x64,0xdc,0x00,0x02,0xe7,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte_d16_hi off, a2, flat_scratch_hi offset:-1

// GFX90A: scratch_store_byte_d16_hi off, a2, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x64,0xdc,0x00,0x02,0xea,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte_d16_hi off, a2, vcc_lo offset:-1

// GFX90A: scratch_store_byte_d16_hi off, a2, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x64,0xdc,0x00,0x02,0xeb,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte_d16_hi off, a2, vcc_hi offset:-1

// GFX90A: scratch_store_byte_d16_hi v0, a2, off offset:-1 ; encoding: [0xff,0x5f,0x64,0xdc,0x00,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte_d16_hi v0, a2, off offset:-1

// GFX90A: scratch_store_byte_d16_hi off, a2, s3 ; encoding: [0x00,0x40,0x64,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte_d16_hi off, a2, s3

// GFX90A: scratch_store_byte_d16_hi off, a2, s3 ; encoding: [0x00,0x40,0x64,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte_d16_hi off, a2, s3

// GFX90A: scratch_store_byte_d16_hi off, a2, s3 offset:4095 ; encoding: [0xff,0x4f,0x64,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte_d16_hi off, a2, s3 offset:4095

// GFX90A: scratch_store_byte_d16_hi off, a2, s3 offset:-4096 ; encoding: [0x00,0x50,0x64,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte_d16_hi off, a2, s3 offset:-4096

// GFX90A: scratch_store_byte_d16_hi off, a2, s3 offset:-1 glc ; encoding: [0xff,0x5f,0x65,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte_d16_hi off, a2, s3 offset:-1 glc

// GFX90A: scratch_store_byte_d16_hi off, a2, s3 offset:-1 slc ; encoding: [0xff,0x5f,0x66,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_byte_d16_hi off, a2, s3 offset:-1 slc

// GFX90A: scratch_store_short off, a2, s3 offset:-1 ; encoding: [0xff,0x5f,0x68,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short off, a2, s3 offset:-1

// GFX90A: scratch_store_short off, a255, s3 offset:-1 ; encoding: [0xff,0x5f,0x68,0xdc,0x00,0xff,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short off, a255, s3 offset:-1

// GFX90A: scratch_store_short off, a2, s101 offset:-1 ; encoding: [0xff,0x5f,0x68,0xdc,0x00,0x02,0xe5,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short off, a2, s101 offset:-1

// GFX90A: scratch_store_short off, a2, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x68,0xdc,0x00,0x02,0xe6,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short off, a2, flat_scratch_lo offset:-1

// GFX90A: scratch_store_short off, a2, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x68,0xdc,0x00,0x02,0xe7,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short off, a2, flat_scratch_hi offset:-1

// GFX90A: scratch_store_short off, a2, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x68,0xdc,0x00,0x02,0xea,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short off, a2, vcc_lo offset:-1

// GFX90A: scratch_store_short off, a2, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x68,0xdc,0x00,0x02,0xeb,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short off, a2, vcc_hi offset:-1

// GFX90A: scratch_store_short v0, a2, off offset:-1 ; encoding: [0xff,0x5f,0x68,0xdc,0x00,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short v0, a2, off offset:-1

// GFX90A: scratch_store_short off, a2, s3 ; encoding: [0x00,0x40,0x68,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short off, a2, s3

// GFX90A: scratch_store_short off, a2, s3 ; encoding: [0x00,0x40,0x68,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short off, a2, s3

// GFX90A: scratch_store_short off, a2, s3 offset:4095 ; encoding: [0xff,0x4f,0x68,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short off, a2, s3 offset:4095

// GFX90A: scratch_store_short off, a2, s3 offset:-4096 ; encoding: [0x00,0x50,0x68,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short off, a2, s3 offset:-4096

// GFX90A: scratch_store_short off, a2, s3 offset:-1 glc ; encoding: [0xff,0x5f,0x69,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short off, a2, s3 offset:-1 glc

// GFX90A: scratch_store_short off, a2, s3 offset:-1 slc ; encoding: [0xff,0x5f,0x6a,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short off, a2, s3 offset:-1 slc

// GFX90A: scratch_store_short_d16_hi off, a2, s3 offset:-1 ; encoding: [0xff,0x5f,0x6c,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short_d16_hi off, a2, s3 offset:-1

// GFX90A: scratch_store_short_d16_hi off, a255, s3 offset:-1 ; encoding: [0xff,0x5f,0x6c,0xdc,0x00,0xff,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short_d16_hi off, a255, s3 offset:-1

// GFX90A: scratch_store_short_d16_hi off, a2, s101 offset:-1 ; encoding: [0xff,0x5f,0x6c,0xdc,0x00,0x02,0xe5,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short_d16_hi off, a2, s101 offset:-1

// GFX90A: scratch_store_short_d16_hi off, a2, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x6c,0xdc,0x00,0x02,0xe6,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short_d16_hi off, a2, flat_scratch_lo offset:-1

// GFX90A: scratch_store_short_d16_hi off, a2, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x6c,0xdc,0x00,0x02,0xe7,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short_d16_hi off, a2, flat_scratch_hi offset:-1

// GFX90A: scratch_store_short_d16_hi off, a2, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x6c,0xdc,0x00,0x02,0xea,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short_d16_hi off, a2, vcc_lo offset:-1

// GFX90A: scratch_store_short_d16_hi off, a2, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x6c,0xdc,0x00,0x02,0xeb,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short_d16_hi off, a2, vcc_hi offset:-1

// GFX90A: scratch_store_short_d16_hi v0, a2, off offset:-1 ; encoding: [0xff,0x5f,0x6c,0xdc,0x00,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short_d16_hi v0, a2, off offset:-1

// GFX90A: scratch_store_short_d16_hi off, a2, s3 ; encoding: [0x00,0x40,0x6c,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short_d16_hi off, a2, s3

// GFX90A: scratch_store_short_d16_hi off, a2, s3 ; encoding: [0x00,0x40,0x6c,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short_d16_hi off, a2, s3

// GFX90A: scratch_store_short_d16_hi off, a2, s3 offset:4095 ; encoding: [0xff,0x4f,0x6c,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short_d16_hi off, a2, s3 offset:4095

// GFX90A: scratch_store_short_d16_hi off, a2, s3 offset:-4096 ; encoding: [0x00,0x50,0x6c,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short_d16_hi off, a2, s3 offset:-4096

// GFX90A: scratch_store_short_d16_hi off, a2, s3 offset:-1 glc ; encoding: [0xff,0x5f,0x6d,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short_d16_hi off, a2, s3 offset:-1 glc

// GFX90A: scratch_store_short_d16_hi off, a2, s3 offset:-1 slc ; encoding: [0xff,0x5f,0x6e,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_short_d16_hi off, a2, s3 offset:-1 slc

// GFX90A: scratch_store_dword off, a2, s3 offset:-1 ; encoding: [0xff,0x5f,0x70,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dword off, a2, s3 offset:-1

// GFX90A: scratch_store_dword off, a255, s3 offset:-1 ; encoding: [0xff,0x5f,0x70,0xdc,0x00,0xff,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dword off, a255, s3 offset:-1

// GFX90A: scratch_store_dword off, a2, s101 offset:-1 ; encoding: [0xff,0x5f,0x70,0xdc,0x00,0x02,0xe5,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dword off, a2, s101 offset:-1

// GFX90A: scratch_store_dword off, a2, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x70,0xdc,0x00,0x02,0xe6,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dword off, a2, flat_scratch_lo offset:-1

// GFX90A: scratch_store_dword off, a2, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x70,0xdc,0x00,0x02,0xe7,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dword off, a2, flat_scratch_hi offset:-1

// GFX90A: scratch_store_dword off, a2, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x70,0xdc,0x00,0x02,0xea,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dword off, a2, vcc_lo offset:-1

// GFX90A: scratch_store_dword off, a2, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x70,0xdc,0x00,0x02,0xeb,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dword off, a2, vcc_hi offset:-1

// GFX90A: scratch_store_dword v0, a2, off offset:-1 ; encoding: [0xff,0x5f,0x70,0xdc,0x00,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dword v0, a2, off offset:-1

// GFX90A: scratch_store_dword off, a2, s3 ; encoding: [0x00,0x40,0x70,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dword off, a2, s3

// GFX90A: scratch_store_dword off, a2, s3 ; encoding: [0x00,0x40,0x70,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dword off, a2, s3

// GFX90A: scratch_store_dword off, a2, s3 offset:4095 ; encoding: [0xff,0x4f,0x70,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dword off, a2, s3 offset:4095

// GFX90A: scratch_store_dword off, a2, s3 offset:-4096 ; encoding: [0x00,0x50,0x70,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dword off, a2, s3 offset:-4096

// GFX90A: scratch_store_dword off, a2, s3 offset:-1 glc ; encoding: [0xff,0x5f,0x71,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dword off, a2, s3 offset:-1 glc

// GFX90A: scratch_store_dword off, a2, s3 offset:-1 slc ; encoding: [0xff,0x5f,0x72,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dword off, a2, s3 offset:-1 slc

// GFX90A: scratch_store_dwordx2 off, a[2:3], s3 offset:-1 ; encoding: [0xff,0x5f,0x74,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx2 off, a[2:3], s3 offset:-1

// GFX90A: scratch_store_dwordx2 off, a[254:255], s3 offset:-1 ; encoding: [0xff,0x5f,0x74,0xdc,0x00,0xfe,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx2 off, a[254:255], s3 offset:-1

// GFX90A: scratch_store_dwordx2 off, a[2:3], s101 offset:-1 ; encoding: [0xff,0x5f,0x74,0xdc,0x00,0x02,0xe5,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx2 off, a[2:3], s101 offset:-1

// GFX90A: scratch_store_dwordx2 off, a[2:3], flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x74,0xdc,0x00,0x02,0xe6,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx2 off, a[2:3], flat_scratch_lo offset:-1

// GFX90A: scratch_store_dwordx2 off, a[2:3], flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x74,0xdc,0x00,0x02,0xe7,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx2 off, a[2:3], flat_scratch_hi offset:-1

// GFX90A: scratch_store_dwordx2 off, a[2:3], vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x74,0xdc,0x00,0x02,0xea,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx2 off, a[2:3], vcc_lo offset:-1

// GFX90A: scratch_store_dwordx2 off, a[2:3], vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x74,0xdc,0x00,0x02,0xeb,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx2 off, a[2:3], vcc_hi offset:-1

// GFX90A: scratch_store_dwordx2 v0, a[2:3], off offset:-1 ; encoding: [0xff,0x5f,0x74,0xdc,0x00,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx2 v0, a[2:3], off offset:-1

// GFX90A: scratch_store_dwordx2 off, a[2:3], s3 ; encoding: [0x00,0x40,0x74,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx2 off, a[2:3], s3

// GFX90A: scratch_store_dwordx2 off, a[2:3], s3 ; encoding: [0x00,0x40,0x74,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx2 off, a[2:3], s3

// GFX90A: scratch_store_dwordx2 off, a[2:3], s3 offset:4095 ; encoding: [0xff,0x4f,0x74,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx2 off, a[2:3], s3 offset:4095

// GFX90A: scratch_store_dwordx2 off, a[2:3], s3 offset:-4096 ; encoding: [0x00,0x50,0x74,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx2 off, a[2:3], s3 offset:-4096

// GFX90A: scratch_store_dwordx2 off, a[2:3], s3 offset:-1 glc ; encoding: [0xff,0x5f,0x75,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx2 off, a[2:3], s3 offset:-1 glc

// GFX90A: scratch_store_dwordx2 off, a[2:3], s3 offset:-1 slc ; encoding: [0xff,0x5f,0x76,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx2 off, a[2:3], s3 offset:-1 slc

// GFX90A: scratch_store_dwordx3 off, a[2:4], s3 offset:-1 ; encoding: [0xff,0x5f,0x78,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx3 off, a[2:4], s3 offset:-1

// GFX90A: scratch_store_dwordx3 off, a[252:254], s3 offset:-1 ; encoding: [0xff,0x5f,0x78,0xdc,0x00,0xfc,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx3 off, a[252:254], s3 offset:-1

// GFX90A: scratch_store_dwordx3 off, a[2:4], s101 offset:-1 ; encoding: [0xff,0x5f,0x78,0xdc,0x00,0x02,0xe5,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx3 off, a[2:4], s101 offset:-1

// GFX90A: scratch_store_dwordx3 off, a[2:4], flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x78,0xdc,0x00,0x02,0xe6,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx3 off, a[2:4], flat_scratch_lo offset:-1

// GFX90A: scratch_store_dwordx3 off, a[2:4], flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x78,0xdc,0x00,0x02,0xe7,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx3 off, a[2:4], flat_scratch_hi offset:-1

// GFX90A: scratch_store_dwordx3 off, a[2:4], vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x78,0xdc,0x00,0x02,0xea,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx3 off, a[2:4], vcc_lo offset:-1

// GFX90A: scratch_store_dwordx3 off, a[2:4], vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x78,0xdc,0x00,0x02,0xeb,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx3 off, a[2:4], vcc_hi offset:-1

// GFX90A: scratch_store_dwordx3 v0, a[2:4], off offset:-1 ; encoding: [0xff,0x5f,0x78,0xdc,0x00,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx3 v0, a[2:4], off offset:-1

// GFX90A: scratch_store_dwordx3 off, a[2:4], s3 ; encoding: [0x00,0x40,0x78,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx3 off, a[2:4], s3

// GFX90A: scratch_store_dwordx3 off, a[2:4], s3 ; encoding: [0x00,0x40,0x78,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx3 off, a[2:4], s3

// GFX90A: scratch_store_dwordx3 off, a[2:4], s3 offset:4095 ; encoding: [0xff,0x4f,0x78,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx3 off, a[2:4], s3 offset:4095

// GFX90A: scratch_store_dwordx3 off, a[2:4], s3 offset:-4096 ; encoding: [0x00,0x50,0x78,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx3 off, a[2:4], s3 offset:-4096

// GFX90A: scratch_store_dwordx3 off, a[2:4], s3 offset:-1 glc ; encoding: [0xff,0x5f,0x79,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx3 off, a[2:4], s3 offset:-1 glc

// GFX90A: scratch_store_dwordx3 off, a[2:4], s3 offset:-1 slc ; encoding: [0xff,0x5f,0x7a,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx3 off, a[2:4], s3 offset:-1 slc

// GFX90A: scratch_store_dwordx4 off, a[2:5], s3 offset:-1 ; encoding: [0xff,0x5f,0x7c,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx4 off, a[2:5], s3 offset:-1

// GFX90A: scratch_store_dwordx4 off, a[252:255], s3 offset:-1 ; encoding: [0xff,0x5f,0x7c,0xdc,0x00,0xfc,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx4 off, a[252:255], s3 offset:-1

// GFX90A: scratch_store_dwordx4 off, a[2:5], s101 offset:-1 ; encoding: [0xff,0x5f,0x7c,0xdc,0x00,0x02,0xe5,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx4 off, a[2:5], s101 offset:-1

// GFX90A: scratch_store_dwordx4 off, a[2:5], flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x7c,0xdc,0x00,0x02,0xe6,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx4 off, a[2:5], flat_scratch_lo offset:-1

// GFX90A: scratch_store_dwordx4 off, a[2:5], flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x7c,0xdc,0x00,0x02,0xe7,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx4 off, a[2:5], flat_scratch_hi offset:-1

// GFX90A: scratch_store_dwordx4 off, a[2:5], vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x7c,0xdc,0x00,0x02,0xea,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx4 off, a[2:5], vcc_lo offset:-1

// GFX90A: scratch_store_dwordx4 off, a[2:5], vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x7c,0xdc,0x00,0x02,0xeb,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx4 off, a[2:5], vcc_hi offset:-1

// GFX90A: scratch_store_dwordx4 v0, a[2:5], off offset:-1 ; encoding: [0xff,0x5f,0x7c,0xdc,0x00,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx4 v0, a[2:5], off offset:-1

// GFX90A: scratch_store_dwordx4 off, a[2:5], s3 ; encoding: [0x00,0x40,0x7c,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx4 off, a[2:5], s3

// GFX90A: scratch_store_dwordx4 off, a[2:5], s3 ; encoding: [0x00,0x40,0x7c,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx4 off, a[2:5], s3

// GFX90A: scratch_store_dwordx4 off, a[2:5], s3 offset:4095 ; encoding: [0xff,0x4f,0x7c,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx4 off, a[2:5], s3 offset:4095

// GFX90A: scratch_store_dwordx4 off, a[2:5], s3 offset:-4096 ; encoding: [0x00,0x50,0x7c,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx4 off, a[2:5], s3 offset:-4096

// GFX90A: scratch_store_dwordx4 off, a[2:5], s3 offset:-1 glc ; encoding: [0xff,0x5f,0x7d,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx4 off, a[2:5], s3 offset:-1 glc

// GFX90A: scratch_store_dwordx4 off, a[2:5], s3 offset:-1 slc ; encoding: [0xff,0x5f,0x7e,0xdc,0x00,0x02,0x83,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_store_dwordx4 off, a[2:5], s3 offset:-1 slc

// GFX90A: scratch_load_ubyte_d16 a5, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x80,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16 a5, off, s2 offset:-1

// GFX90A: scratch_load_ubyte_d16 a255, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x80,0xdc,0x00,0x00,0x82,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16 a255, off, s2 offset:-1

// GFX90A: scratch_load_ubyte_d16 a5, off, s101 offset:-1 ; encoding: [0xff,0x5f,0x80,0xdc,0x00,0x00,0xe5,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16 a5, off, s101 offset:-1

// GFX90A: scratch_load_ubyte_d16 a5, off, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x80,0xdc,0x00,0x00,0xe6,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16 a5, off, flat_scratch_lo offset:-1

// GFX90A: scratch_load_ubyte_d16 a5, off, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x80,0xdc,0x00,0x00,0xe7,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16 a5, off, flat_scratch_hi offset:-1

// GFX90A: scratch_load_ubyte_d16 a5, off, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x80,0xdc,0x00,0x00,0xea,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16 a5, off, vcc_lo offset:-1

// GFX90A: scratch_load_ubyte_d16 a5, off, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x80,0xdc,0x00,0x00,0xeb,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16 a5, off, vcc_hi offset:-1

// GFX90A: scratch_load_ubyte_d16 a5, v0, off offset:-1 ; encoding: [0xff,0x5f,0x80,0xdc,0x00,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16 a5, v0, off offset:-1

// GFX90A: scratch_load_ubyte_d16 a5, off, s2 ; encoding: [0x00,0x40,0x80,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16 a5, off, s2

// GFX90A: scratch_load_ubyte_d16 a5, off, s2 ; encoding: [0x00,0x40,0x80,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16 a5, off, s2

// GFX90A: scratch_load_ubyte_d16 a5, off, s2 offset:4095 ; encoding: [0xff,0x4f,0x80,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16 a5, off, s2 offset:4095

// GFX90A: scratch_load_ubyte_d16 a5, off, s2 offset:-4096 ; encoding: [0x00,0x50,0x80,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16 a5, off, s2 offset:-4096

// GFX90A: scratch_load_ubyte_d16 a5, off, s2 offset:-1 glc ; encoding: [0xff,0x5f,0x81,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16 a5, off, s2 offset:-1 glc

// GFX90A: scratch_load_ubyte_d16 a5, off, s2 offset:-1 slc ; encoding: [0xff,0x5f,0x82,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16 a5, off, s2 offset:-1 slc

// GFX90A: scratch_load_ubyte_d16_hi a5, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x84,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16_hi a5, off, s2 offset:-1

// GFX90A: scratch_load_ubyte_d16_hi a255, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x84,0xdc,0x00,0x00,0x82,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16_hi a255, off, s2 offset:-1

// GFX90A: scratch_load_ubyte_d16_hi a5, off, s101 offset:-1 ; encoding: [0xff,0x5f,0x84,0xdc,0x00,0x00,0xe5,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16_hi a5, off, s101 offset:-1

// GFX90A: scratch_load_ubyte_d16_hi a5, off, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x84,0xdc,0x00,0x00,0xe6,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16_hi a5, off, flat_scratch_lo offset:-1

// GFX90A: scratch_load_ubyte_d16_hi a5, off, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x84,0xdc,0x00,0x00,0xe7,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16_hi a5, off, flat_scratch_hi offset:-1

// GFX90A: scratch_load_ubyte_d16_hi a5, off, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x84,0xdc,0x00,0x00,0xea,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16_hi a5, off, vcc_lo offset:-1

// GFX90A: scratch_load_ubyte_d16_hi a5, off, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x84,0xdc,0x00,0x00,0xeb,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16_hi a5, off, vcc_hi offset:-1

// GFX90A: scratch_load_ubyte_d16_hi a5, v0, off offset:-1 ; encoding: [0xff,0x5f,0x84,0xdc,0x00,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16_hi a5, v0, off offset:-1

// GFX90A: scratch_load_ubyte_d16_hi a5, off, s2 ; encoding: [0x00,0x40,0x84,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16_hi a5, off, s2

// GFX90A: scratch_load_ubyte_d16_hi a5, off, s2 ; encoding: [0x00,0x40,0x84,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16_hi a5, off, s2

// GFX90A: scratch_load_ubyte_d16_hi a5, off, s2 offset:4095 ; encoding: [0xff,0x4f,0x84,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16_hi a5, off, s2 offset:4095

// GFX90A: scratch_load_ubyte_d16_hi a5, off, s2 offset:-4096 ; encoding: [0x00,0x50,0x84,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16_hi a5, off, s2 offset:-4096

// GFX90A: scratch_load_ubyte_d16_hi a5, off, s2 offset:-1 glc ; encoding: [0xff,0x5f,0x85,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16_hi a5, off, s2 offset:-1 glc

// GFX90A: scratch_load_ubyte_d16_hi a5, off, s2 offset:-1 slc ; encoding: [0xff,0x5f,0x86,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_ubyte_d16_hi a5, off, s2 offset:-1 slc

// GFX90A: scratch_load_sbyte_d16 a5, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x88,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16 a5, off, s2 offset:-1

// GFX90A: scratch_load_sbyte_d16 a255, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x88,0xdc,0x00,0x00,0x82,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16 a255, off, s2 offset:-1

// GFX90A: scratch_load_sbyte_d16 a5, off, s101 offset:-1 ; encoding: [0xff,0x5f,0x88,0xdc,0x00,0x00,0xe5,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16 a5, off, s101 offset:-1

// GFX90A: scratch_load_sbyte_d16 a5, off, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x88,0xdc,0x00,0x00,0xe6,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16 a5, off, flat_scratch_lo offset:-1

// GFX90A: scratch_load_sbyte_d16 a5, off, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x88,0xdc,0x00,0x00,0xe7,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16 a5, off, flat_scratch_hi offset:-1

// GFX90A: scratch_load_sbyte_d16 a5, off, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x88,0xdc,0x00,0x00,0xea,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16 a5, off, vcc_lo offset:-1

// GFX90A: scratch_load_sbyte_d16 a5, off, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x88,0xdc,0x00,0x00,0xeb,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16 a5, off, vcc_hi offset:-1

// GFX90A: scratch_load_sbyte_d16 a5, v0, off offset:-1 ; encoding: [0xff,0x5f,0x88,0xdc,0x00,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16 a5, v0, off offset:-1

// GFX90A: scratch_load_sbyte_d16 a5, off, s2 ; encoding: [0x00,0x40,0x88,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16 a5, off, s2

// GFX90A: scratch_load_sbyte_d16 a5, off, s2 ; encoding: [0x00,0x40,0x88,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16 a5, off, s2

// GFX90A: scratch_load_sbyte_d16 a5, off, s2 offset:4095 ; encoding: [0xff,0x4f,0x88,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16 a5, off, s2 offset:4095

// GFX90A: scratch_load_sbyte_d16 a5, off, s2 offset:-4096 ; encoding: [0x00,0x50,0x88,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16 a5, off, s2 offset:-4096

// GFX90A: scratch_load_sbyte_d16 a5, off, s2 offset:-1 glc ; encoding: [0xff,0x5f,0x89,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16 a5, off, s2 offset:-1 glc

// GFX90A: scratch_load_sbyte_d16 a5, off, s2 offset:-1 slc ; encoding: [0xff,0x5f,0x8a,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16 a5, off, s2 offset:-1 slc

// GFX90A: scratch_load_sbyte_d16_hi a5, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x8c,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16_hi a5, off, s2 offset:-1

// GFX90A: scratch_load_sbyte_d16_hi a255, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x8c,0xdc,0x00,0x00,0x82,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16_hi a255, off, s2 offset:-1

// GFX90A: scratch_load_sbyte_d16_hi a5, off, s101 offset:-1 ; encoding: [0xff,0x5f,0x8c,0xdc,0x00,0x00,0xe5,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16_hi a5, off, s101 offset:-1

// GFX90A: scratch_load_sbyte_d16_hi a5, off, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x8c,0xdc,0x00,0x00,0xe6,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16_hi a5, off, flat_scratch_lo offset:-1

// GFX90A: scratch_load_sbyte_d16_hi a5, off, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x8c,0xdc,0x00,0x00,0xe7,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16_hi a5, off, flat_scratch_hi offset:-1

// GFX90A: scratch_load_sbyte_d16_hi a5, off, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x8c,0xdc,0x00,0x00,0xea,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16_hi a5, off, vcc_lo offset:-1

// GFX90A: scratch_load_sbyte_d16_hi a5, off, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x8c,0xdc,0x00,0x00,0xeb,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16_hi a5, off, vcc_hi offset:-1

// GFX90A: scratch_load_sbyte_d16_hi a5, v0, off offset:-1 ; encoding: [0xff,0x5f,0x8c,0xdc,0x00,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16_hi a5, v0, off offset:-1

// GFX90A: scratch_load_sbyte_d16_hi a5, off, s2 ; encoding: [0x00,0x40,0x8c,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16_hi a5, off, s2

// GFX90A: scratch_load_sbyte_d16_hi a5, off, s2 ; encoding: [0x00,0x40,0x8c,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16_hi a5, off, s2

// GFX90A: scratch_load_sbyte_d16_hi a5, off, s2 offset:4095 ; encoding: [0xff,0x4f,0x8c,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16_hi a5, off, s2 offset:4095

// GFX90A: scratch_load_sbyte_d16_hi a5, off, s2 offset:-4096 ; encoding: [0x00,0x50,0x8c,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16_hi a5, off, s2 offset:-4096

// GFX90A: scratch_load_sbyte_d16_hi a5, off, s2 offset:-1 glc ; encoding: [0xff,0x5f,0x8d,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16_hi a5, off, s2 offset:-1 glc

// GFX90A: scratch_load_sbyte_d16_hi a5, off, s2 offset:-1 slc ; encoding: [0xff,0x5f,0x8e,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_sbyte_d16_hi a5, off, s2 offset:-1 slc

// GFX90A: scratch_load_short_d16 a5, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x90,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16 a5, off, s2 offset:-1

// GFX90A: scratch_load_short_d16 a255, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x90,0xdc,0x00,0x00,0x82,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16 a255, off, s2 offset:-1

// GFX90A: scratch_load_short_d16 a5, off, s101 offset:-1 ; encoding: [0xff,0x5f,0x90,0xdc,0x00,0x00,0xe5,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16 a5, off, s101 offset:-1

// GFX90A: scratch_load_short_d16 a5, off, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x90,0xdc,0x00,0x00,0xe6,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16 a5, off, flat_scratch_lo offset:-1

// GFX90A: scratch_load_short_d16 a5, off, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x90,0xdc,0x00,0x00,0xe7,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16 a5, off, flat_scratch_hi offset:-1

// GFX90A: scratch_load_short_d16 a5, off, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x90,0xdc,0x00,0x00,0xea,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16 a5, off, vcc_lo offset:-1

// GFX90A: scratch_load_short_d16 a5, off, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x90,0xdc,0x00,0x00,0xeb,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16 a5, off, vcc_hi offset:-1

// GFX90A: scratch_load_short_d16 a5, v0, off offset:-1 ; encoding: [0xff,0x5f,0x90,0xdc,0x00,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16 a5, v0, off offset:-1

// GFX90A: scratch_load_short_d16 a5, off, s2 ; encoding: [0x00,0x40,0x90,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16 a5, off, s2

// GFX90A: scratch_load_short_d16 a5, off, s2 ; encoding: [0x00,0x40,0x90,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16 a5, off, s2

// GFX90A: scratch_load_short_d16 a5, off, s2 offset:4095 ; encoding: [0xff,0x4f,0x90,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16 a5, off, s2 offset:4095

// GFX90A: scratch_load_short_d16 a5, off, s2 offset:-4096 ; encoding: [0x00,0x50,0x90,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16 a5, off, s2 offset:-4096

// GFX90A: scratch_load_short_d16 a5, off, s2 offset:-1 glc ; encoding: [0xff,0x5f,0x91,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16 a5, off, s2 offset:-1 glc

// GFX90A: scratch_load_short_d16 a5, off, s2 offset:-1 slc ; encoding: [0xff,0x5f,0x92,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16 a5, off, s2 offset:-1 slc

// GFX90A: scratch_load_short_d16_hi a5, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x94,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16_hi a5, off, s2 offset:-1

// GFX90A: scratch_load_short_d16_hi a255, off, s2 offset:-1 ; encoding: [0xff,0x5f,0x94,0xdc,0x00,0x00,0x82,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16_hi a255, off, s2 offset:-1

// GFX90A: scratch_load_short_d16_hi a5, off, s101 offset:-1 ; encoding: [0xff,0x5f,0x94,0xdc,0x00,0x00,0xe5,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16_hi a5, off, s101 offset:-1

// GFX90A: scratch_load_short_d16_hi a5, off, flat_scratch_lo offset:-1 ; encoding: [0xff,0x5f,0x94,0xdc,0x00,0x00,0xe6,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16_hi a5, off, flat_scratch_lo offset:-1

// GFX90A: scratch_load_short_d16_hi a5, off, flat_scratch_hi offset:-1 ; encoding: [0xff,0x5f,0x94,0xdc,0x00,0x00,0xe7,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16_hi a5, off, flat_scratch_hi offset:-1

// GFX90A: scratch_load_short_d16_hi a5, off, vcc_lo offset:-1 ; encoding: [0xff,0x5f,0x94,0xdc,0x00,0x00,0xea,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16_hi a5, off, vcc_lo offset:-1

// GFX90A: scratch_load_short_d16_hi a5, off, vcc_hi offset:-1 ; encoding: [0xff,0x5f,0x94,0xdc,0x00,0x00,0xeb,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16_hi a5, off, vcc_hi offset:-1

// GFX90A: scratch_load_short_d16_hi a5, v0, off offset:-1 ; encoding: [0xff,0x5f,0x94,0xdc,0x00,0x00,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16_hi a5, v0, off offset:-1

// GFX90A: scratch_load_short_d16_hi a5, off, s2 ; encoding: [0x00,0x40,0x94,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16_hi a5, off, s2

// GFX90A: scratch_load_short_d16_hi a5, off, s2 ; encoding: [0x00,0x40,0x94,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16_hi a5, off, s2

// GFX90A: scratch_load_short_d16_hi a5, off, s2 offset:4095 ; encoding: [0xff,0x4f,0x94,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16_hi a5, off, s2 offset:4095

// GFX90A: scratch_load_short_d16_hi a5, off, s2 offset:-4096 ; encoding: [0x00,0x50,0x94,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16_hi a5, off, s2 offset:-4096

// GFX90A: scratch_load_short_d16_hi a5, off, s2 offset:-1 glc ; encoding: [0xff,0x5f,0x95,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16_hi a5, off, s2 offset:-1 glc

// GFX90A: scratch_load_short_d16_hi a5, off, s2 offset:-1 slc ; encoding: [0xff,0x5f,0x96,0xdc,0x00,0x00,0x82,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
scratch_load_short_d16_hi a5, off, s2 offset:-1 slc

// GFX90A: buffer_load_format_x a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_format_x a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe0,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_format_x a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe0,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_format_x a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe0,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_format_x a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe0,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_format_x a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe0,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_format_x a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe0,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_format_x a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe0,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_format_x a5, off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe0,0x00,0x05,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a5, off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_format_x a5, off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe0,0x00,0x05,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a5, off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_format_x a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x00,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_format_x a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x00,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_format_x a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x00,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a5, off, s[8:11], s3

// GFX90A: buffer_load_format_x a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x00,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a5, off, s[8:11], s3

// GFX90A: buffer_load_format_x a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x00,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_load_format_x a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x00,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_format_x a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x02,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_x a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_format_xy a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_format_xy a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe0,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_format_xy a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe0,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_format_xy a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe0,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_format_xy a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe0,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_format_xy a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe0,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_format_xy a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe0,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_format_xy a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe0,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_format_xy a[6:7], off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe0,0x00,0x06,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[6:7], off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_format_xy a[6:7], off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe0,0x00,0x06,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[6:7], off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_format_xy a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x04,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_format_xy a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x04,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_format_xy a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x04,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[6:7], off, s[8:11], s3

// GFX90A: buffer_load_format_xy a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x04,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[6:7], off, s[8:11], s3

// GFX90A: buffer_load_format_xy a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x04,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_load_format_xy a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x04,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_format_xy a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x06,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xy a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_format_xyz a[6:8], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[6:8], off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_format_xyz a[252:254], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe0,0x00,0xfc,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[252:254], off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_format_xyz a[6:8], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe0,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[6:8], off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_format_xyz a[6:8], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe0,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[6:8], off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_format_xyz a[6:8], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe0,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[6:8], off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_format_xyz a[6:8], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe0,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[6:8], off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_format_xyz a[6:8], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe0,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[6:8], off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_format_xyz a[6:8], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe0,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[6:8], off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_format_xyz a[6:8], off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe0,0x00,0x06,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[6:8], off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_format_xyz a[6:8], off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe0,0x00,0x06,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[6:8], off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_format_xyz a[6:8], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x08,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[6:8], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_format_xyz a[6:8], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x08,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[6:8], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_format_xyz a[6:8], off, s[8:11], s3 ; encoding: [0x00,0x00,0x08,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[6:8], off, s[8:11], s3

// GFX90A: buffer_load_format_xyz a[6:8], off, s[8:11], s3 ; encoding: [0x00,0x00,0x08,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[6:8], off, s[8:11], s3

// GFX90A: buffer_load_format_xyz a[6:8], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x08,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[6:8], off, s[8:11], s3 offset:7

// GFX90A: buffer_load_format_xyz a[6:8], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x08,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[6:8], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_format_xyz a[6:8], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x0a,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyz a[6:8], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_format_xyzw a[6:9], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[6:9], off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_format_xyzw a[252:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe0,0x00,0xfc,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[252:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_format_xyzw a[6:9], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe0,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[6:9], off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_format_xyzw a[6:9], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe0,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[6:9], off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_format_xyzw a[6:9], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe0,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[6:9], off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_format_xyzw a[6:9], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe0,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[6:9], off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_format_xyzw a[6:9], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe0,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[6:9], off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_format_xyzw a[6:9], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe0,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[6:9], off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_format_xyzw a[6:9], off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe0,0x00,0x06,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[6:9], off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_format_xyzw a[6:9], off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe0,0x00,0x06,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[6:9], off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_format_xyzw a[6:9], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x0c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[6:9], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_format_xyzw a[6:9], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x0c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[6:9], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_format_xyzw a[6:9], off, s[8:11], s3 ; encoding: [0x00,0x00,0x0c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[6:9], off, s[8:11], s3

// GFX90A: buffer_load_format_xyzw a[6:9], off, s[8:11], s3 ; encoding: [0x00,0x00,0x0c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[6:9], off, s[8:11], s3

// GFX90A: buffer_load_format_xyzw a[6:9], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x0c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[6:9], off, s[8:11], s3 offset:7

// GFX90A: buffer_load_format_xyzw a[6:9], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x0c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[6:9], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_format_xyzw a[6:9], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x0e,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_xyzw a[6:9], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_store_format_x a1, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a1, off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_format_x a255, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe0,0x00,0xff,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a255, off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_format_x a1, off, s[16:19], s4 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe0,0x00,0x01,0x84,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a1, off, s[16:19], s4 offset:4095

// GFX90A: buffer_store_format_x a1, off, s[96:99], s4 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe0,0x00,0x01,0x98,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a1, off, s[96:99], s4 offset:4095

// GFX90A: buffer_store_format_x a1, off, s[12:15], s101 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe0,0x00,0x01,0x83,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a1, off, s[12:15], s101 offset:4095

// GFX90A: buffer_store_format_x a1, off, s[12:15], m0 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe0,0x00,0x01,0x83,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a1, off, s[12:15], m0 offset:4095

// GFX90A: buffer_store_format_x a1, off, s[12:15], 0 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe0,0x00,0x01,0x83,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a1, off, s[12:15], 0 offset:4095

// GFX90A: buffer_store_format_x a1, off, s[12:15], -1 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe0,0x00,0x01,0x83,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a1, off, s[12:15], -1 offset:4095

// GFX90A: buffer_store_format_x a1, off, s[12:15], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe0,0x00,0x01,0x83,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a1, off, s[12:15], 0.5 offset:4095

// GFX90A: buffer_store_format_x a1, off, s[12:15], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe0,0x00,0x01,0x83,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a1, off, s[12:15], -4.0 offset:4095

// GFX90A: buffer_store_format_x a1, v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x10,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a1, v0, s[12:15], s4 idxen offset:4095

// GFX90A: buffer_store_format_x a1, v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x10,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a1, v0, s[12:15], s4 offen offset:4095

// GFX90A: buffer_store_format_x a1, off, s[12:15], s4 ; encoding: [0x00,0x00,0x10,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a1, off, s[12:15], s4

// GFX90A: buffer_store_format_x a1, off, s[12:15], s4 ; encoding: [0x00,0x00,0x10,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a1, off, s[12:15], s4

// GFX90A: buffer_store_format_x a1, off, s[12:15], s4 offset:7 ; encoding: [0x07,0x00,0x10,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a1, off, s[12:15], s4 offset:7

// GFX90A: buffer_store_format_x a1, off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x10,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a1, off, s[12:15], s4 offset:4095 glc

// GFX90A: buffer_store_format_x a1, off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x12,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_x a1, off, s[12:15], s4 offset:4095 slc

// GFX90A: buffer_store_format_xy a[2:3], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[2:3], off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_format_xy a[254:255], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe0,0x00,0xfe,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[254:255], off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_format_xy a[2:3], off, s[16:19], s4 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe0,0x00,0x02,0x84,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[2:3], off, s[16:19], s4 offset:4095

// GFX90A: buffer_store_format_xy a[2:3], off, s[96:99], s4 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe0,0x00,0x02,0x98,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[2:3], off, s[96:99], s4 offset:4095

// GFX90A: buffer_store_format_xy a[2:3], off, s[12:15], s101 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe0,0x00,0x02,0x83,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[2:3], off, s[12:15], s101 offset:4095

// GFX90A: buffer_store_format_xy a[2:3], off, s[12:15], m0 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe0,0x00,0x02,0x83,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[2:3], off, s[12:15], m0 offset:4095

// GFX90A: buffer_store_format_xy a[2:3], off, s[12:15], 0 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe0,0x00,0x02,0x83,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[2:3], off, s[12:15], 0 offset:4095

// GFX90A: buffer_store_format_xy a[2:3], off, s[12:15], -1 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe0,0x00,0x02,0x83,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[2:3], off, s[12:15], -1 offset:4095

// GFX90A: buffer_store_format_xy a[2:3], off, s[12:15], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe0,0x00,0x02,0x83,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[2:3], off, s[12:15], 0.5 offset:4095

// GFX90A: buffer_store_format_xy a[2:3], off, s[12:15], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe0,0x00,0x02,0x83,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[2:3], off, s[12:15], -4.0 offset:4095

// GFX90A: buffer_store_format_xy a[2:3], v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x14,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[2:3], v0, s[12:15], s4 idxen offset:4095

// GFX90A: buffer_store_format_xy a[2:3], v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x14,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[2:3], v0, s[12:15], s4 offen offset:4095

// GFX90A: buffer_store_format_xy a[2:3], off, s[12:15], s4 ; encoding: [0x00,0x00,0x14,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[2:3], off, s[12:15], s4

// GFX90A: buffer_store_format_xy a[2:3], off, s[12:15], s4 ; encoding: [0x00,0x00,0x14,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[2:3], off, s[12:15], s4

// GFX90A: buffer_store_format_xy a[2:3], off, s[12:15], s4 offset:7 ; encoding: [0x07,0x00,0x14,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[2:3], off, s[12:15], s4 offset:7

// GFX90A: buffer_store_format_xy a[2:3], off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x14,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[2:3], off, s[12:15], s4 offset:4095 glc

// GFX90A: buffer_store_format_xy a[2:3], off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x16,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xy a[2:3], off, s[12:15], s4 offset:4095 slc

// GFX90A: buffer_store_format_xyz a[2:4], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[2:4], off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_format_xyz a[252:254], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe0,0x00,0xfc,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[252:254], off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_format_xyz a[2:4], off, s[16:19], s4 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe0,0x00,0x02,0x84,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[2:4], off, s[16:19], s4 offset:4095

// GFX90A: buffer_store_format_xyz a[2:4], off, s[96:99], s4 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe0,0x00,0x02,0x98,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[2:4], off, s[96:99], s4 offset:4095

// GFX90A: buffer_store_format_xyz a[2:4], off, s[12:15], s101 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe0,0x00,0x02,0x83,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[2:4], off, s[12:15], s101 offset:4095

// GFX90A: buffer_store_format_xyz a[2:4], off, s[12:15], m0 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe0,0x00,0x02,0x83,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[2:4], off, s[12:15], m0 offset:4095

// GFX90A: buffer_store_format_xyz a[2:4], off, s[12:15], 0 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe0,0x00,0x02,0x83,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[2:4], off, s[12:15], 0 offset:4095

// GFX90A: buffer_store_format_xyz a[2:4], off, s[12:15], -1 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe0,0x00,0x02,0x83,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[2:4], off, s[12:15], -1 offset:4095

// GFX90A: buffer_store_format_xyz a[2:4], off, s[12:15], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe0,0x00,0x02,0x83,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[2:4], off, s[12:15], 0.5 offset:4095

// GFX90A: buffer_store_format_xyz a[2:4], off, s[12:15], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe0,0x00,0x02,0x83,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[2:4], off, s[12:15], -4.0 offset:4095

// GFX90A: buffer_store_format_xyz a[2:4], v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x18,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[2:4], v0, s[12:15], s4 idxen offset:4095

// GFX90A: buffer_store_format_xyz a[2:4], v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x18,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[2:4], v0, s[12:15], s4 offen offset:4095

// GFX90A: buffer_store_format_xyz a[2:4], off, s[12:15], s4 ; encoding: [0x00,0x00,0x18,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[2:4], off, s[12:15], s4

// GFX90A: buffer_store_format_xyz a[2:4], off, s[12:15], s4 ; encoding: [0x00,0x00,0x18,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[2:4], off, s[12:15], s4

// GFX90A: buffer_store_format_xyz a[2:4], off, s[12:15], s4 offset:7 ; encoding: [0x07,0x00,0x18,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[2:4], off, s[12:15], s4 offset:7

// GFX90A: buffer_store_format_xyz a[2:4], off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x18,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[2:4], off, s[12:15], s4 offset:4095 glc

// GFX90A: buffer_store_format_xyz a[2:4], off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x1a,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyz a[2:4], off, s[12:15], s4 offset:4095 slc

// GFX90A: buffer_store_format_xyzw a[2:5], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[2:5], off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_format_xyzw a[252:255], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe0,0x00,0xfc,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[252:255], off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_format_xyzw a[2:5], off, s[16:19], s4 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe0,0x00,0x02,0x84,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[2:5], off, s[16:19], s4 offset:4095

// GFX90A: buffer_store_format_xyzw a[2:5], off, s[96:99], s4 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe0,0x00,0x02,0x98,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[2:5], off, s[96:99], s4 offset:4095

// GFX90A: buffer_store_format_xyzw a[2:5], off, s[12:15], s101 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe0,0x00,0x02,0x83,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[2:5], off, s[12:15], s101 offset:4095

// GFX90A: buffer_store_format_xyzw a[2:5], off, s[12:15], m0 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe0,0x00,0x02,0x83,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[2:5], off, s[12:15], m0 offset:4095

// GFX90A: buffer_store_format_xyzw a[2:5], off, s[12:15], 0 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe0,0x00,0x02,0x83,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[2:5], off, s[12:15], 0 offset:4095

// GFX90A: buffer_store_format_xyzw a[2:5], off, s[12:15], -1 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe0,0x00,0x02,0x83,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[2:5], off, s[12:15], -1 offset:4095

// GFX90A: buffer_store_format_xyzw a[2:5], off, s[12:15], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe0,0x00,0x02,0x83,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[2:5], off, s[12:15], 0.5 offset:4095

// GFX90A: buffer_store_format_xyzw a[2:5], off, s[12:15], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe0,0x00,0x02,0x83,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[2:5], off, s[12:15], -4.0 offset:4095

// GFX90A: buffer_store_format_xyzw a[2:5], v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x1c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[2:5], v0, s[12:15], s4 idxen offset:4095

// GFX90A: buffer_store_format_xyzw a[2:5], v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x1c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[2:5], v0, s[12:15], s4 offen offset:4095

// GFX90A: buffer_store_format_xyzw a[2:5], off, s[12:15], s4 ; encoding: [0x00,0x00,0x1c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[2:5], off, s[12:15], s4

// GFX90A: buffer_store_format_xyzw a[2:5], off, s[12:15], s4 ; encoding: [0x00,0x00,0x1c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[2:5], off, s[12:15], s4

// GFX90A: buffer_store_format_xyzw a[2:5], off, s[12:15], s4 offset:7 ; encoding: [0x07,0x00,0x1c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[2:5], off, s[12:15], s4 offset:7

// GFX90A: buffer_store_format_xyzw a[2:5], off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x1c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[2:5], off, s[12:15], s4 offset:4095 glc

// GFX90A: buffer_store_format_xyzw a[2:5], off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x1e,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_xyzw a[2:5], off, s[12:15], s4 offset:4095 slc

// GFX90A: buffer_load_format_d16_x a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_format_d16_x a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe0,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_format_d16_x a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe0,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_format_d16_x a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe0,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_format_d16_x a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe0,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_format_d16_x a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe0,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_format_d16_x a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe0,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_format_d16_x a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe0,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_format_d16_x a5, off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe0,0x00,0x05,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a5, off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_format_d16_x a5, off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe0,0x00,0x05,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a5, off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_format_d16_x a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x20,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_format_d16_x a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x20,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_format_d16_x a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x20,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a5, off, s[8:11], s3

// GFX90A: buffer_load_format_d16_x a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x20,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a5, off, s[8:11], s3

// GFX90A: buffer_load_format_d16_x a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x20,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_load_format_d16_x a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x20,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_format_d16_x a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x22,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_x a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_format_d16_xy a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_format_d16_xy a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe0,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_format_d16_xy a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe0,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_format_d16_xy a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe0,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_format_d16_xy a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe0,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_format_d16_xy a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe0,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_format_d16_xy a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe0,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_format_d16_xy a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe0,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_format_d16_xy a5, off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe0,0x00,0x05,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a5, off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_format_d16_xy a5, off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe0,0x00,0x05,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a5, off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_format_d16_xy a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x24,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_format_d16_xy a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x24,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_format_d16_xy a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x24,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a5, off, s[8:11], s3

// GFX90A: buffer_load_format_d16_xy a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x24,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a5, off, s[8:11], s3

// GFX90A: buffer_load_format_d16_xy a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x24,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_load_format_d16_xy a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x24,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_format_d16_xy a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x26,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xy a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_format_d16_xyz a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_format_d16_xyz a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe0,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_format_d16_xyz a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe0,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_format_d16_xyz a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe0,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_format_d16_xyz a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe0,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_format_d16_xyz a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe0,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_format_d16_xyz a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe0,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_format_d16_xyz a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe0,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_format_d16_xyz a[6:7], off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe0,0x00,0x06,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[6:7], off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_format_d16_xyz a[6:7], off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe0,0x00,0x06,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[6:7], off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_format_d16_xyz a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x28,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_format_d16_xyz a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x28,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_format_d16_xyz a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x28,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[6:7], off, s[8:11], s3

// GFX90A: buffer_load_format_d16_xyz a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x28,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[6:7], off, s[8:11], s3

// GFX90A: buffer_load_format_d16_xyz a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x28,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_load_format_d16_xyz a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x28,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_format_d16_xyz a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x2a,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyz a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_format_d16_xyzw a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_format_d16_xyzw a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe0,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_format_d16_xyzw a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe0,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_format_d16_xyzw a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe0,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_format_d16_xyzw a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe0,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_format_d16_xyzw a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe0,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_format_d16_xyzw a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe0,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_format_d16_xyzw a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe0,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_format_d16_xyzw a[6:7], off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe0,0x00,0x06,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[6:7], off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_format_d16_xyzw a[6:7], off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe0,0x00,0x06,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[6:7], off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_format_d16_xyzw a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x2c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_format_d16_xyzw a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x2c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_format_d16_xyzw a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x2c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[6:7], off, s[8:11], s3

// GFX90A: buffer_load_format_d16_xyzw a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x2c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[6:7], off, s[8:11], s3

// GFX90A: buffer_load_format_d16_xyzw a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x2c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_load_format_d16_xyzw a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x2c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_format_d16_xyzw a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x2e,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_format_d16_xyzw a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_store_format_d16_x a1, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a1, off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_format_d16_x a255, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe0,0x00,0xff,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a255, off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_format_d16_x a1, off, s[16:19], s4 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe0,0x00,0x01,0x84,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a1, off, s[16:19], s4 offset:4095

// GFX90A: buffer_store_format_d16_x a1, off, s[96:99], s4 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe0,0x00,0x01,0x98,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a1, off, s[96:99], s4 offset:4095

// GFX90A: buffer_store_format_d16_x a1, off, s[12:15], s101 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe0,0x00,0x01,0x83,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a1, off, s[12:15], s101 offset:4095

// GFX90A: buffer_store_format_d16_x a1, off, s[12:15], m0 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe0,0x00,0x01,0x83,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a1, off, s[12:15], m0 offset:4095

// GFX90A: buffer_store_format_d16_x a1, off, s[12:15], 0 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe0,0x00,0x01,0x83,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a1, off, s[12:15], 0 offset:4095

// GFX90A: buffer_store_format_d16_x a1, off, s[12:15], -1 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe0,0x00,0x01,0x83,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a1, off, s[12:15], -1 offset:4095

// GFX90A: buffer_store_format_d16_x a1, off, s[12:15], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe0,0x00,0x01,0x83,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a1, off, s[12:15], 0.5 offset:4095

// GFX90A: buffer_store_format_d16_x a1, off, s[12:15], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe0,0x00,0x01,0x83,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a1, off, s[12:15], -4.0 offset:4095

// GFX90A: buffer_store_format_d16_x a1, v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x30,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a1, v0, s[12:15], s4 idxen offset:4095

// GFX90A: buffer_store_format_d16_x a1, v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x30,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a1, v0, s[12:15], s4 offen offset:4095

// GFX90A: buffer_store_format_d16_x a1, off, s[12:15], s4 ; encoding: [0x00,0x00,0x30,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a1, off, s[12:15], s4

// GFX90A: buffer_store_format_d16_x a1, off, s[12:15], s4 ; encoding: [0x00,0x00,0x30,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a1, off, s[12:15], s4

// GFX90A: buffer_store_format_d16_x a1, off, s[12:15], s4 offset:7 ; encoding: [0x07,0x00,0x30,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a1, off, s[12:15], s4 offset:7

// GFX90A: buffer_store_format_d16_x a1, off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x30,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a1, off, s[12:15], s4 offset:4095 glc

// GFX90A: buffer_store_format_d16_x a1, off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x32,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_x a1, off, s[12:15], s4 offset:4095 slc

// GFX90A: buffer_store_format_d16_xy a1, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x34,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a1, off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_format_d16_xy a255, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x34,0xe0,0x00,0xff,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a255, off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_format_d16_xy a1, off, s[16:19], s4 offset:4095 ; encoding: [0xff,0x0f,0x34,0xe0,0x00,0x01,0x84,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a1, off, s[16:19], s4 offset:4095

// GFX90A: buffer_store_format_d16_xy a1, off, s[96:99], s4 offset:4095 ; encoding: [0xff,0x0f,0x34,0xe0,0x00,0x01,0x98,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a1, off, s[96:99], s4 offset:4095

// GFX90A: buffer_store_format_d16_xy a1, off, s[12:15], s101 offset:4095 ; encoding: [0xff,0x0f,0x34,0xe0,0x00,0x01,0x83,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a1, off, s[12:15], s101 offset:4095

// GFX90A: buffer_store_format_d16_xy a1, off, s[12:15], m0 offset:4095 ; encoding: [0xff,0x0f,0x34,0xe0,0x00,0x01,0x83,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a1, off, s[12:15], m0 offset:4095

// GFX90A: buffer_store_format_d16_xy a1, off, s[12:15], 0 offset:4095 ; encoding: [0xff,0x0f,0x34,0xe0,0x00,0x01,0x83,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a1, off, s[12:15], 0 offset:4095

// GFX90A: buffer_store_format_d16_xy a1, off, s[12:15], -1 offset:4095 ; encoding: [0xff,0x0f,0x34,0xe0,0x00,0x01,0x83,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a1, off, s[12:15], -1 offset:4095

// GFX90A: buffer_store_format_d16_xy a1, off, s[12:15], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x34,0xe0,0x00,0x01,0x83,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a1, off, s[12:15], 0.5 offset:4095

// GFX90A: buffer_store_format_d16_xy a1, off, s[12:15], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x34,0xe0,0x00,0x01,0x83,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a1, off, s[12:15], -4.0 offset:4095

// GFX90A: buffer_store_format_d16_xy a1, v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x34,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a1, v0, s[12:15], s4 idxen offset:4095

// GFX90A: buffer_store_format_d16_xy a1, v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x34,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a1, v0, s[12:15], s4 offen offset:4095

// GFX90A: buffer_store_format_d16_xy a1, off, s[12:15], s4 ; encoding: [0x00,0x00,0x34,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a1, off, s[12:15], s4

// GFX90A: buffer_store_format_d16_xy a1, off, s[12:15], s4 ; encoding: [0x00,0x00,0x34,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a1, off, s[12:15], s4

// GFX90A: buffer_store_format_d16_xy a1, off, s[12:15], s4 offset:7 ; encoding: [0x07,0x00,0x34,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a1, off, s[12:15], s4 offset:7

// GFX90A: buffer_store_format_d16_xy a1, off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x34,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a1, off, s[12:15], s4 offset:4095 glc

// GFX90A: buffer_store_format_d16_xy a1, off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x36,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xy a1, off, s[12:15], s4 offset:4095 slc

// GFX90A: buffer_store_format_d16_xyz a[2:3], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x38,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[2:3], off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_format_d16_xyz a[254:255], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x38,0xe0,0x00,0xfe,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[254:255], off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_format_d16_xyz a[2:3], off, s[16:19], s4 offset:4095 ; encoding: [0xff,0x0f,0x38,0xe0,0x00,0x02,0x84,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[2:3], off, s[16:19], s4 offset:4095

// GFX90A: buffer_store_format_d16_xyz a[2:3], off, s[96:99], s4 offset:4095 ; encoding: [0xff,0x0f,0x38,0xe0,0x00,0x02,0x98,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[2:3], off, s[96:99], s4 offset:4095

// GFX90A: buffer_store_format_d16_xyz a[2:3], off, s[12:15], s101 offset:4095 ; encoding: [0xff,0x0f,0x38,0xe0,0x00,0x02,0x83,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[2:3], off, s[12:15], s101 offset:4095

// GFX90A: buffer_store_format_d16_xyz a[2:3], off, s[12:15], m0 offset:4095 ; encoding: [0xff,0x0f,0x38,0xe0,0x00,0x02,0x83,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[2:3], off, s[12:15], m0 offset:4095

// GFX90A: buffer_store_format_d16_xyz a[2:3], off, s[12:15], 0 offset:4095 ; encoding: [0xff,0x0f,0x38,0xe0,0x00,0x02,0x83,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[2:3], off, s[12:15], 0 offset:4095

// GFX90A: buffer_store_format_d16_xyz a[2:3], off, s[12:15], -1 offset:4095 ; encoding: [0xff,0x0f,0x38,0xe0,0x00,0x02,0x83,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[2:3], off, s[12:15], -1 offset:4095

// GFX90A: buffer_store_format_d16_xyz a[2:3], off, s[12:15], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x38,0xe0,0x00,0x02,0x83,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[2:3], off, s[12:15], 0.5 offset:4095

// GFX90A: buffer_store_format_d16_xyz a[2:3], off, s[12:15], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x38,0xe0,0x00,0x02,0x83,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[2:3], off, s[12:15], -4.0 offset:4095

// GFX90A: buffer_store_format_d16_xyz a[2:3], v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x38,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[2:3], v0, s[12:15], s4 idxen offset:4095

// GFX90A: buffer_store_format_d16_xyz a[2:3], v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x38,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[2:3], v0, s[12:15], s4 offen offset:4095

// GFX90A: buffer_store_format_d16_xyz a[2:3], off, s[12:15], s4 ; encoding: [0x00,0x00,0x38,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[2:3], off, s[12:15], s4

// GFX90A: buffer_store_format_d16_xyz a[2:3], off, s[12:15], s4 ; encoding: [0x00,0x00,0x38,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[2:3], off, s[12:15], s4

// GFX90A: buffer_store_format_d16_xyz a[2:3], off, s[12:15], s4 offset:7 ; encoding: [0x07,0x00,0x38,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[2:3], off, s[12:15], s4 offset:7

// GFX90A: buffer_store_format_d16_xyz a[2:3], off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x38,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[2:3], off, s[12:15], s4 offset:4095 glc

// GFX90A: buffer_store_format_d16_xyz a[2:3], off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x3a,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyz a[2:3], off, s[12:15], s4 offset:4095 slc

// GFX90A: buffer_store_format_d16_xyzw a[2:3], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[2:3], off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_format_d16_xyzw a[254:255], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe0,0x00,0xfe,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[254:255], off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_format_d16_xyzw a[2:3], off, s[16:19], s4 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe0,0x00,0x02,0x84,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[2:3], off, s[16:19], s4 offset:4095

// GFX90A: buffer_store_format_d16_xyzw a[2:3], off, s[96:99], s4 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe0,0x00,0x02,0x98,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[2:3], off, s[96:99], s4 offset:4095

// GFX90A: buffer_store_format_d16_xyzw a[2:3], off, s[12:15], s101 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe0,0x00,0x02,0x83,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[2:3], off, s[12:15], s101 offset:4095

// GFX90A: buffer_store_format_d16_xyzw a[2:3], off, s[12:15], m0 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe0,0x00,0x02,0x83,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[2:3], off, s[12:15], m0 offset:4095

// GFX90A: buffer_store_format_d16_xyzw a[2:3], off, s[12:15], 0 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe0,0x00,0x02,0x83,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[2:3], off, s[12:15], 0 offset:4095

// GFX90A: buffer_store_format_d16_xyzw a[2:3], off, s[12:15], -1 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe0,0x00,0x02,0x83,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[2:3], off, s[12:15], -1 offset:4095

// GFX90A: buffer_store_format_d16_xyzw a[2:3], off, s[12:15], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe0,0x00,0x02,0x83,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[2:3], off, s[12:15], 0.5 offset:4095

// GFX90A: buffer_store_format_d16_xyzw a[2:3], off, s[12:15], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x3c,0xe0,0x00,0x02,0x83,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[2:3], off, s[12:15], -4.0 offset:4095

// GFX90A: buffer_store_format_d16_xyzw a[2:3], v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x3c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[2:3], v0, s[12:15], s4 idxen offset:4095

// GFX90A: buffer_store_format_d16_xyzw a[2:3], v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x3c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[2:3], v0, s[12:15], s4 offen offset:4095

// GFX90A: buffer_store_format_d16_xyzw a[2:3], off, s[12:15], s4 ; encoding: [0x00,0x00,0x3c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[2:3], off, s[12:15], s4

// GFX90A: buffer_store_format_d16_xyzw a[2:3], off, s[12:15], s4 ; encoding: [0x00,0x00,0x3c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[2:3], off, s[12:15], s4

// GFX90A: buffer_store_format_d16_xyzw a[2:3], off, s[12:15], s4 offset:7 ; encoding: [0x07,0x00,0x3c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[2:3], off, s[12:15], s4 offset:7

// GFX90A: buffer_store_format_d16_xyzw a[2:3], off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x3c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[2:3], off, s[12:15], s4 offset:4095 glc

// GFX90A: buffer_store_format_d16_xyzw a[2:3], off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x3e,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_format_d16_xyzw a[2:3], off, s[12:15], s4 offset:4095 slc

// GFX90A: buffer_load_ubyte a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_ubyte a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe0,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_ubyte a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe0,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_ubyte a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe0,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_ubyte a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe0,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_ubyte a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe0,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_ubyte a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe0,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_ubyte a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe0,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_ubyte a5, off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe0,0x00,0x05,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a5, off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_ubyte a5, off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x40,0xe0,0x00,0x05,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a5, off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_ubyte a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x40,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_ubyte a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x40,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_ubyte a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x40,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a5, off, s[8:11], s3

// GFX90A: buffer_load_ubyte a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x40,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a5, off, s[8:11], s3

// GFX90A: buffer_load_ubyte a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x40,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_load_ubyte a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x40,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_ubyte a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x42,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_sbyte a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_sbyte a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe0,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_sbyte a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe0,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_sbyte a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe0,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_sbyte a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe0,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_sbyte a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe0,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_sbyte a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe0,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_sbyte a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe0,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_sbyte a5, off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe0,0x00,0x05,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a5, off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_sbyte a5, off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x44,0xe0,0x00,0x05,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a5, off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_sbyte a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x44,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_sbyte a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x44,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_sbyte a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x44,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a5, off, s[8:11], s3

// GFX90A: buffer_load_sbyte a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x44,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a5, off, s[8:11], s3

// GFX90A: buffer_load_sbyte a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x44,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_load_sbyte a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x44,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_sbyte a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x46,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_ushort a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_ushort a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe0,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_ushort a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe0,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_ushort a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe0,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_ushort a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe0,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_ushort a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe0,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_ushort a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe0,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_ushort a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe0,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_ushort a5, off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe0,0x00,0x05,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a5, off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_ushort a5, off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x48,0xe0,0x00,0x05,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a5, off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_ushort a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x48,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_ushort a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x48,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_ushort a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x48,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a5, off, s[8:11], s3

// GFX90A: buffer_load_ushort a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x48,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a5, off, s[8:11], s3

// GFX90A: buffer_load_ushort a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x48,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_load_ushort a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x48,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_ushort a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x4a,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ushort a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_sshort a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x4c,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_sshort a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x4c,0xe0,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_sshort a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x4c,0xe0,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_sshort a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x4c,0xe0,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_sshort a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x4c,0xe0,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_sshort a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x4c,0xe0,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_sshort a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x4c,0xe0,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_sshort a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x4c,0xe0,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_sshort a5, off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x4c,0xe0,0x00,0x05,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a5, off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_sshort a5, off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x4c,0xe0,0x00,0x05,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a5, off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_sshort a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x4c,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_sshort a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x4c,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_sshort a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x4c,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a5, off, s[8:11], s3

// GFX90A: buffer_load_sshort a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x4c,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a5, off, s[8:11], s3

// GFX90A: buffer_load_sshort a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x4c,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_load_sshort a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x4c,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_sshort a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x4e,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sshort a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_dword a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x50,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_dword a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x50,0xe0,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_dword a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x50,0xe0,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_dword a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x50,0xe0,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_dword a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x50,0xe0,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_dword a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x50,0xe0,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_dword a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x50,0xe0,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_dword a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x50,0xe0,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_dword a5, off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x50,0xe0,0x00,0x05,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a5, off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_dword a5, off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x50,0xe0,0x00,0x05,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a5, off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_dword a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x50,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_dword a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x50,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_dword a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x50,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a5, off, s[8:11], s3

// GFX90A: buffer_load_dword a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x50,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a5, off, s[8:11], s3

// GFX90A: buffer_load_dword a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x50,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_load_dword a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x50,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_dword a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x52,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dword a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_dwordx2 a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x54,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_dwordx2 a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x54,0xe0,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_dwordx2 a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x54,0xe0,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_dwordx2 a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x54,0xe0,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_dwordx2 a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x54,0xe0,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_dwordx2 a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x54,0xe0,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_dwordx2 a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x54,0xe0,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_dwordx2 a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x54,0xe0,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_dwordx2 a[6:7], off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x54,0xe0,0x00,0x06,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[6:7], off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_dwordx2 a[6:7], off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x54,0xe0,0x00,0x06,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[6:7], off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_dwordx2 a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x54,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_dwordx2 a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x54,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_dwordx2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x54,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_load_dwordx2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x54,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_load_dwordx2 a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x54,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_load_dwordx2 a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x54,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_dwordx2 a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x56,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx2 a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_dwordx3 a[6:8], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x58,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[6:8], off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_dwordx3 a[252:254], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x58,0xe0,0x00,0xfc,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[252:254], off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_dwordx3 a[6:8], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x58,0xe0,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[6:8], off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_dwordx3 a[6:8], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x58,0xe0,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[6:8], off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_dwordx3 a[6:8], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x58,0xe0,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[6:8], off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_dwordx3 a[6:8], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x58,0xe0,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[6:8], off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_dwordx3 a[6:8], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x58,0xe0,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[6:8], off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_dwordx3 a[6:8], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x58,0xe0,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[6:8], off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_dwordx3 a[6:8], off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x58,0xe0,0x00,0x06,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[6:8], off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_dwordx3 a[6:8], off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x58,0xe0,0x00,0x06,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[6:8], off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_dwordx3 a[6:8], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x58,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[6:8], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_dwordx3 a[6:8], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x58,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[6:8], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_dwordx3 a[6:8], off, s[8:11], s3 ; encoding: [0x00,0x00,0x58,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[6:8], off, s[8:11], s3

// GFX90A: buffer_load_dwordx3 a[6:8], off, s[8:11], s3 ; encoding: [0x00,0x00,0x58,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[6:8], off, s[8:11], s3

// GFX90A: buffer_load_dwordx3 a[6:8], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x58,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[6:8], off, s[8:11], s3 offset:7

// GFX90A: buffer_load_dwordx3 a[6:8], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x58,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[6:8], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_dwordx3 a[6:8], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x5a,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx3 a[6:8], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_dwordx4 a[6:9], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x5c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[6:9], off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_dwordx4 a[252:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x5c,0xe0,0x00,0xfc,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[252:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_dwordx4 a[6:9], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x5c,0xe0,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[6:9], off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_dwordx4 a[6:9], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x5c,0xe0,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[6:9], off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_dwordx4 a[6:9], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x5c,0xe0,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[6:9], off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_dwordx4 a[6:9], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x5c,0xe0,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[6:9], off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_dwordx4 a[6:9], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x5c,0xe0,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[6:9], off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_dwordx4 a[6:9], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x5c,0xe0,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[6:9], off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_dwordx4 a[6:9], off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x5c,0xe0,0x00,0x06,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[6:9], off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_dwordx4 a[6:9], off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x5c,0xe0,0x00,0x06,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[6:9], off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_dwordx4 a[6:9], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x5c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[6:9], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_dwordx4 a[6:9], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x5c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[6:9], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_dwordx4 a[6:9], off, s[8:11], s3 ; encoding: [0x00,0x00,0x5c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[6:9], off, s[8:11], s3

// GFX90A: buffer_load_dwordx4 a[6:9], off, s[8:11], s3 ; encoding: [0x00,0x00,0x5c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[6:9], off, s[8:11], s3

// GFX90A: buffer_load_dwordx4 a[6:9], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x5c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[6:9], off, s[8:11], s3 offset:7

// GFX90A: buffer_load_dwordx4 a[6:9], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x5c,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[6:9], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_dwordx4 a[6:9], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x5e,0xe0,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_dwordx4 a[6:9], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_store_byte a1, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x60,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a1, off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_byte a255, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x60,0xe0,0x00,0xff,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a255, off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_byte a1, off, s[16:19], s4 offset:4095 ; encoding: [0xff,0x0f,0x60,0xe0,0x00,0x01,0x84,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a1, off, s[16:19], s4 offset:4095

// GFX90A: buffer_store_byte a1, off, s[96:99], s4 offset:4095 ; encoding: [0xff,0x0f,0x60,0xe0,0x00,0x01,0x98,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a1, off, s[96:99], s4 offset:4095

// GFX90A: buffer_store_byte a1, off, s[12:15], s101 offset:4095 ; encoding: [0xff,0x0f,0x60,0xe0,0x00,0x01,0x83,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a1, off, s[12:15], s101 offset:4095

// GFX90A: buffer_store_byte a1, off, s[12:15], m0 offset:4095 ; encoding: [0xff,0x0f,0x60,0xe0,0x00,0x01,0x83,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a1, off, s[12:15], m0 offset:4095

// GFX90A: buffer_store_byte a1, off, s[12:15], 0 offset:4095 ; encoding: [0xff,0x0f,0x60,0xe0,0x00,0x01,0x83,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a1, off, s[12:15], 0 offset:4095

// GFX90A: buffer_store_byte a1, off, s[12:15], -1 offset:4095 ; encoding: [0xff,0x0f,0x60,0xe0,0x00,0x01,0x83,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a1, off, s[12:15], -1 offset:4095

// GFX90A: buffer_store_byte a1, off, s[12:15], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x60,0xe0,0x00,0x01,0x83,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a1, off, s[12:15], 0.5 offset:4095

// GFX90A: buffer_store_byte a1, off, s[12:15], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x60,0xe0,0x00,0x01,0x83,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a1, off, s[12:15], -4.0 offset:4095

// GFX90A: buffer_store_byte a1, v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x60,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a1, v0, s[12:15], s4 idxen offset:4095

// GFX90A: buffer_store_byte a1, v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x60,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a1, v0, s[12:15], s4 offen offset:4095

// GFX90A: buffer_store_byte a1, off, s[12:15], s4 ; encoding: [0x00,0x00,0x60,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a1, off, s[12:15], s4

// GFX90A: buffer_store_byte a1, off, s[12:15], s4 ; encoding: [0x00,0x00,0x60,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a1, off, s[12:15], s4

// GFX90A: buffer_store_byte a1, off, s[12:15], s4 offset:7 ; encoding: [0x07,0x00,0x60,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a1, off, s[12:15], s4 offset:7

// GFX90A: buffer_store_byte a1, off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x60,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a1, off, s[12:15], s4 offset:4095 glc

// GFX90A: buffer_store_byte a1, off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x62,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte a1, off, s[12:15], s4 offset:4095 slc

// GFX90A: buffer_store_byte_d16_hi a1, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x64,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a1, off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_byte_d16_hi a255, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x64,0xe0,0x00,0xff,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a255, off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_byte_d16_hi a1, off, s[16:19], s4 offset:4095 ; encoding: [0xff,0x0f,0x64,0xe0,0x00,0x01,0x84,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a1, off, s[16:19], s4 offset:4095

// GFX90A: buffer_store_byte_d16_hi a1, off, s[96:99], s4 offset:4095 ; encoding: [0xff,0x0f,0x64,0xe0,0x00,0x01,0x98,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a1, off, s[96:99], s4 offset:4095

// GFX90A: buffer_store_byte_d16_hi a1, off, s[12:15], s101 offset:4095 ; encoding: [0xff,0x0f,0x64,0xe0,0x00,0x01,0x83,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a1, off, s[12:15], s101 offset:4095

// GFX90A: buffer_store_byte_d16_hi a1, off, s[12:15], m0 offset:4095 ; encoding: [0xff,0x0f,0x64,0xe0,0x00,0x01,0x83,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a1, off, s[12:15], m0 offset:4095

// GFX90A: buffer_store_byte_d16_hi a1, off, s[12:15], 0 offset:4095 ; encoding: [0xff,0x0f,0x64,0xe0,0x00,0x01,0x83,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a1, off, s[12:15], 0 offset:4095

// GFX90A: buffer_store_byte_d16_hi a1, off, s[12:15], -1 offset:4095 ; encoding: [0xff,0x0f,0x64,0xe0,0x00,0x01,0x83,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a1, off, s[12:15], -1 offset:4095

// GFX90A: buffer_store_byte_d16_hi a1, off, s[12:15], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x64,0xe0,0x00,0x01,0x83,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a1, off, s[12:15], 0.5 offset:4095

// GFX90A: buffer_store_byte_d16_hi a1, off, s[12:15], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x64,0xe0,0x00,0x01,0x83,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a1, off, s[12:15], -4.0 offset:4095

// GFX90A: buffer_store_byte_d16_hi a1, v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x64,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a1, v0, s[12:15], s4 idxen offset:4095

// GFX90A: buffer_store_byte_d16_hi a1, v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x64,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a1, v0, s[12:15], s4 offen offset:4095

// GFX90A: buffer_store_byte_d16_hi a1, off, s[12:15], s4 ; encoding: [0x00,0x00,0x64,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a1, off, s[12:15], s4

// GFX90A: buffer_store_byte_d16_hi a1, off, s[12:15], s4 ; encoding: [0x00,0x00,0x64,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a1, off, s[12:15], s4

// GFX90A: buffer_store_byte_d16_hi a1, off, s[12:15], s4 offset:7 ; encoding: [0x07,0x00,0x64,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a1, off, s[12:15], s4 offset:7

// GFX90A: buffer_store_byte_d16_hi a1, off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x64,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a1, off, s[12:15], s4 offset:4095 glc

// GFX90A: buffer_store_byte_d16_hi a1, off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x66,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_byte_d16_hi a1, off, s[12:15], s4 offset:4095 slc

// GFX90A: buffer_store_short a1, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x68,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a1, off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_short a255, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x68,0xe0,0x00,0xff,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a255, off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_short a1, off, s[16:19], s4 offset:4095 ; encoding: [0xff,0x0f,0x68,0xe0,0x00,0x01,0x84,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a1, off, s[16:19], s4 offset:4095

// GFX90A: buffer_store_short a1, off, s[96:99], s4 offset:4095 ; encoding: [0xff,0x0f,0x68,0xe0,0x00,0x01,0x98,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a1, off, s[96:99], s4 offset:4095

// GFX90A: buffer_store_short a1, off, s[12:15], s101 offset:4095 ; encoding: [0xff,0x0f,0x68,0xe0,0x00,0x01,0x83,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a1, off, s[12:15], s101 offset:4095

// GFX90A: buffer_store_short a1, off, s[12:15], m0 offset:4095 ; encoding: [0xff,0x0f,0x68,0xe0,0x00,0x01,0x83,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a1, off, s[12:15], m0 offset:4095

// GFX90A: buffer_store_short a1, off, s[12:15], 0 offset:4095 ; encoding: [0xff,0x0f,0x68,0xe0,0x00,0x01,0x83,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a1, off, s[12:15], 0 offset:4095

// GFX90A: buffer_store_short a1, off, s[12:15], -1 offset:4095 ; encoding: [0xff,0x0f,0x68,0xe0,0x00,0x01,0x83,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a1, off, s[12:15], -1 offset:4095

// GFX90A: buffer_store_short a1, off, s[12:15], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x68,0xe0,0x00,0x01,0x83,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a1, off, s[12:15], 0.5 offset:4095

// GFX90A: buffer_store_short a1, off, s[12:15], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x68,0xe0,0x00,0x01,0x83,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a1, off, s[12:15], -4.0 offset:4095

// GFX90A: buffer_store_short a1, v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x68,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a1, v0, s[12:15], s4 idxen offset:4095

// GFX90A: buffer_store_short a1, v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x68,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a1, v0, s[12:15], s4 offen offset:4095

// GFX90A: buffer_store_short a1, off, s[12:15], s4 ; encoding: [0x00,0x00,0x68,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a1, off, s[12:15], s4

// GFX90A: buffer_store_short a1, off, s[12:15], s4 ; encoding: [0x00,0x00,0x68,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a1, off, s[12:15], s4

// GFX90A: buffer_store_short a1, off, s[12:15], s4 offset:7 ; encoding: [0x07,0x00,0x68,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a1, off, s[12:15], s4 offset:7

// GFX90A: buffer_store_short a1, off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x68,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a1, off, s[12:15], s4 offset:4095 glc

// GFX90A: buffer_store_short a1, off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x6a,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short a1, off, s[12:15], s4 offset:4095 slc

// GFX90A: buffer_store_short_d16_hi a1, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x6c,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a1, off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_short_d16_hi a255, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x6c,0xe0,0x00,0xff,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a255, off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_short_d16_hi a1, off, s[16:19], s4 offset:4095 ; encoding: [0xff,0x0f,0x6c,0xe0,0x00,0x01,0x84,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a1, off, s[16:19], s4 offset:4095

// GFX90A: buffer_store_short_d16_hi a1, off, s[96:99], s4 offset:4095 ; encoding: [0xff,0x0f,0x6c,0xe0,0x00,0x01,0x98,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a1, off, s[96:99], s4 offset:4095

// GFX90A: buffer_store_short_d16_hi a1, off, s[12:15], s101 offset:4095 ; encoding: [0xff,0x0f,0x6c,0xe0,0x00,0x01,0x83,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a1, off, s[12:15], s101 offset:4095

// GFX90A: buffer_store_short_d16_hi a1, off, s[12:15], m0 offset:4095 ; encoding: [0xff,0x0f,0x6c,0xe0,0x00,0x01,0x83,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a1, off, s[12:15], m0 offset:4095

// GFX90A: buffer_store_short_d16_hi a1, off, s[12:15], 0 offset:4095 ; encoding: [0xff,0x0f,0x6c,0xe0,0x00,0x01,0x83,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a1, off, s[12:15], 0 offset:4095

// GFX90A: buffer_store_short_d16_hi a1, off, s[12:15], -1 offset:4095 ; encoding: [0xff,0x0f,0x6c,0xe0,0x00,0x01,0x83,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a1, off, s[12:15], -1 offset:4095

// GFX90A: buffer_store_short_d16_hi a1, off, s[12:15], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x6c,0xe0,0x00,0x01,0x83,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a1, off, s[12:15], 0.5 offset:4095

// GFX90A: buffer_store_short_d16_hi a1, off, s[12:15], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x6c,0xe0,0x00,0x01,0x83,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a1, off, s[12:15], -4.0 offset:4095

// GFX90A: buffer_store_short_d16_hi a1, v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x6c,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a1, v0, s[12:15], s4 idxen offset:4095

// GFX90A: buffer_store_short_d16_hi a1, v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x6c,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a1, v0, s[12:15], s4 offen offset:4095

// GFX90A: buffer_store_short_d16_hi a1, off, s[12:15], s4 ; encoding: [0x00,0x00,0x6c,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a1, off, s[12:15], s4

// GFX90A: buffer_store_short_d16_hi a1, off, s[12:15], s4 ; encoding: [0x00,0x00,0x6c,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a1, off, s[12:15], s4

// GFX90A: buffer_store_short_d16_hi a1, off, s[12:15], s4 offset:7 ; encoding: [0x07,0x00,0x6c,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a1, off, s[12:15], s4 offset:7

// GFX90A: buffer_store_short_d16_hi a1, off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x6c,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a1, off, s[12:15], s4 offset:4095 glc

// GFX90A: buffer_store_short_d16_hi a1, off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x6e,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_short_d16_hi a1, off, s[12:15], s4 offset:4095 slc

// GFX90A: buffer_store_dword a1, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x70,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a1, off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_dword a255, off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x70,0xe0,0x00,0xff,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a255, off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_dword a1, off, s[16:19], s4 offset:4095 ; encoding: [0xff,0x0f,0x70,0xe0,0x00,0x01,0x84,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a1, off, s[16:19], s4 offset:4095

// GFX90A: buffer_store_dword a1, off, s[96:99], s4 offset:4095 ; encoding: [0xff,0x0f,0x70,0xe0,0x00,0x01,0x98,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a1, off, s[96:99], s4 offset:4095

// GFX90A: buffer_store_dword a1, off, s[12:15], s101 offset:4095 ; encoding: [0xff,0x0f,0x70,0xe0,0x00,0x01,0x83,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a1, off, s[12:15], s101 offset:4095

// GFX90A: buffer_store_dword a1, off, s[12:15], m0 offset:4095 ; encoding: [0xff,0x0f,0x70,0xe0,0x00,0x01,0x83,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a1, off, s[12:15], m0 offset:4095

// GFX90A: buffer_store_dword a1, off, s[12:15], 0 offset:4095 ; encoding: [0xff,0x0f,0x70,0xe0,0x00,0x01,0x83,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a1, off, s[12:15], 0 offset:4095

// GFX90A: buffer_store_dword a1, off, s[12:15], -1 offset:4095 ; encoding: [0xff,0x0f,0x70,0xe0,0x00,0x01,0x83,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a1, off, s[12:15], -1 offset:4095

// GFX90A: buffer_store_dword a1, off, s[12:15], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x70,0xe0,0x00,0x01,0x83,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a1, off, s[12:15], 0.5 offset:4095

// GFX90A: buffer_store_dword a1, off, s[12:15], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x70,0xe0,0x00,0x01,0x83,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a1, off, s[12:15], -4.0 offset:4095

// GFX90A: buffer_store_dword a1, v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x70,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a1, v0, s[12:15], s4 idxen offset:4095

// GFX90A: buffer_store_dword a1, v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x70,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a1, v0, s[12:15], s4 offen offset:4095

// GFX90A: buffer_store_dword a1, off, s[12:15], s4 ; encoding: [0x00,0x00,0x70,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a1, off, s[12:15], s4

// GFX90A: buffer_store_dword a1, off, s[12:15], s4 ; encoding: [0x00,0x00,0x70,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a1, off, s[12:15], s4

// GFX90A: buffer_store_dword a1, off, s[12:15], s4 offset:7 ; encoding: [0x07,0x00,0x70,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a1, off, s[12:15], s4 offset:7

// GFX90A: buffer_store_dword a1, off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x70,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a1, off, s[12:15], s4 offset:4095 glc

// GFX90A: buffer_store_dword a1, off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x72,0xe0,0x00,0x01,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dword a1, off, s[12:15], s4 offset:4095 slc

// GFX90A: buffer_store_dwordx2 a[2:3], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x74,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[2:3], off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_dwordx2 a[254:255], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x74,0xe0,0x00,0xfe,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[254:255], off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_dwordx2 a[2:3], off, s[16:19], s4 offset:4095 ; encoding: [0xff,0x0f,0x74,0xe0,0x00,0x02,0x84,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[2:3], off, s[16:19], s4 offset:4095

// GFX90A: buffer_store_dwordx2 a[2:3], off, s[96:99], s4 offset:4095 ; encoding: [0xff,0x0f,0x74,0xe0,0x00,0x02,0x98,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[2:3], off, s[96:99], s4 offset:4095

// GFX90A: buffer_store_dwordx2 a[2:3], off, s[12:15], s101 offset:4095 ; encoding: [0xff,0x0f,0x74,0xe0,0x00,0x02,0x83,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[2:3], off, s[12:15], s101 offset:4095

// GFX90A: buffer_store_dwordx2 a[2:3], off, s[12:15], m0 offset:4095 ; encoding: [0xff,0x0f,0x74,0xe0,0x00,0x02,0x83,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[2:3], off, s[12:15], m0 offset:4095

// GFX90A: buffer_store_dwordx2 a[2:3], off, s[12:15], 0 offset:4095 ; encoding: [0xff,0x0f,0x74,0xe0,0x00,0x02,0x83,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[2:3], off, s[12:15], 0 offset:4095

// GFX90A: buffer_store_dwordx2 a[2:3], off, s[12:15], -1 offset:4095 ; encoding: [0xff,0x0f,0x74,0xe0,0x00,0x02,0x83,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[2:3], off, s[12:15], -1 offset:4095

// GFX90A: buffer_store_dwordx2 a[2:3], off, s[12:15], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x74,0xe0,0x00,0x02,0x83,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[2:3], off, s[12:15], 0.5 offset:4095

// GFX90A: buffer_store_dwordx2 a[2:3], off, s[12:15], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x74,0xe0,0x00,0x02,0x83,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[2:3], off, s[12:15], -4.0 offset:4095

// GFX90A: buffer_store_dwordx2 a[2:3], v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x74,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[2:3], v0, s[12:15], s4 idxen offset:4095

// GFX90A: buffer_store_dwordx2 a[2:3], v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x74,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[2:3], v0, s[12:15], s4 offen offset:4095

// GFX90A: buffer_store_dwordx2 a[2:3], off, s[12:15], s4 ; encoding: [0x00,0x00,0x74,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[2:3], off, s[12:15], s4

// GFX90A: buffer_store_dwordx2 a[2:3], off, s[12:15], s4 ; encoding: [0x00,0x00,0x74,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[2:3], off, s[12:15], s4

// GFX90A: buffer_store_dwordx2 a[2:3], off, s[12:15], s4 offset:7 ; encoding: [0x07,0x00,0x74,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[2:3], off, s[12:15], s4 offset:7

// GFX90A: buffer_store_dwordx2 a[2:3], off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x74,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[2:3], off, s[12:15], s4 offset:4095 glc

// GFX90A: buffer_store_dwordx2 a[2:3], off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x76,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx2 a[2:3], off, s[12:15], s4 offset:4095 slc

// GFX90A: buffer_store_dwordx3 a[2:4], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x78,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[2:4], off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_dwordx3 a[252:254], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x78,0xe0,0x00,0xfc,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[252:254], off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_dwordx3 a[2:4], off, s[16:19], s4 offset:4095 ; encoding: [0xff,0x0f,0x78,0xe0,0x00,0x02,0x84,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[2:4], off, s[16:19], s4 offset:4095

// GFX90A: buffer_store_dwordx3 a[2:4], off, s[96:99], s4 offset:4095 ; encoding: [0xff,0x0f,0x78,0xe0,0x00,0x02,0x98,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[2:4], off, s[96:99], s4 offset:4095

// GFX90A: buffer_store_dwordx3 a[2:4], off, s[12:15], s101 offset:4095 ; encoding: [0xff,0x0f,0x78,0xe0,0x00,0x02,0x83,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[2:4], off, s[12:15], s101 offset:4095

// GFX90A: buffer_store_dwordx3 a[2:4], off, s[12:15], m0 offset:4095 ; encoding: [0xff,0x0f,0x78,0xe0,0x00,0x02,0x83,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[2:4], off, s[12:15], m0 offset:4095

// GFX90A: buffer_store_dwordx3 a[2:4], off, s[12:15], 0 offset:4095 ; encoding: [0xff,0x0f,0x78,0xe0,0x00,0x02,0x83,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[2:4], off, s[12:15], 0 offset:4095

// GFX90A: buffer_store_dwordx3 a[2:4], off, s[12:15], -1 offset:4095 ; encoding: [0xff,0x0f,0x78,0xe0,0x00,0x02,0x83,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[2:4], off, s[12:15], -1 offset:4095

// GFX90A: buffer_store_dwordx3 a[2:4], off, s[12:15], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x78,0xe0,0x00,0x02,0x83,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[2:4], off, s[12:15], 0.5 offset:4095

// GFX90A: buffer_store_dwordx3 a[2:4], off, s[12:15], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x78,0xe0,0x00,0x02,0x83,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[2:4], off, s[12:15], -4.0 offset:4095

// GFX90A: buffer_store_dwordx3 a[2:4], v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x78,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[2:4], v0, s[12:15], s4 idxen offset:4095

// GFX90A: buffer_store_dwordx3 a[2:4], v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x78,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[2:4], v0, s[12:15], s4 offen offset:4095

// GFX90A: buffer_store_dwordx3 a[2:4], off, s[12:15], s4 ; encoding: [0x00,0x00,0x78,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[2:4], off, s[12:15], s4

// GFX90A: buffer_store_dwordx3 a[2:4], off, s[12:15], s4 ; encoding: [0x00,0x00,0x78,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[2:4], off, s[12:15], s4

// GFX90A: buffer_store_dwordx3 a[2:4], off, s[12:15], s4 offset:7 ; encoding: [0x07,0x00,0x78,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[2:4], off, s[12:15], s4 offset:7

// GFX90A: buffer_store_dwordx3 a[2:4], off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x78,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[2:4], off, s[12:15], s4 offset:4095 glc

// GFX90A: buffer_store_dwordx3 a[2:4], off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x7a,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx3 a[2:4], off, s[12:15], s4 offset:4095 slc

// GFX90A: buffer_store_dwordx4 a[2:5], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x7c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[2:5], off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_dwordx4 a[252:255], off, s[12:15], s4 offset:4095 ; encoding: [0xff,0x0f,0x7c,0xe0,0x00,0xfc,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[252:255], off, s[12:15], s4 offset:4095

// GFX90A: buffer_store_dwordx4 a[2:5], off, s[16:19], s4 offset:4095 ; encoding: [0xff,0x0f,0x7c,0xe0,0x00,0x02,0x84,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[2:5], off, s[16:19], s4 offset:4095

// GFX90A: buffer_store_dwordx4 a[2:5], off, s[96:99], s4 offset:4095 ; encoding: [0xff,0x0f,0x7c,0xe0,0x00,0x02,0x98,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[2:5], off, s[96:99], s4 offset:4095

// GFX90A: buffer_store_dwordx4 a[2:5], off, s[12:15], s101 offset:4095 ; encoding: [0xff,0x0f,0x7c,0xe0,0x00,0x02,0x83,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[2:5], off, s[12:15], s101 offset:4095

// GFX90A: buffer_store_dwordx4 a[2:5], off, s[12:15], m0 offset:4095 ; encoding: [0xff,0x0f,0x7c,0xe0,0x00,0x02,0x83,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[2:5], off, s[12:15], m0 offset:4095

// GFX90A: buffer_store_dwordx4 a[2:5], off, s[12:15], 0 offset:4095 ; encoding: [0xff,0x0f,0x7c,0xe0,0x00,0x02,0x83,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[2:5], off, s[12:15], 0 offset:4095

// GFX90A: buffer_store_dwordx4 a[2:5], off, s[12:15], -1 offset:4095 ; encoding: [0xff,0x0f,0x7c,0xe0,0x00,0x02,0x83,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[2:5], off, s[12:15], -1 offset:4095

// GFX90A: buffer_store_dwordx4 a[2:5], off, s[12:15], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x7c,0xe0,0x00,0x02,0x83,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[2:5], off, s[12:15], 0.5 offset:4095

// GFX90A: buffer_store_dwordx4 a[2:5], off, s[12:15], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x7c,0xe0,0x00,0x02,0x83,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[2:5], off, s[12:15], -4.0 offset:4095

// GFX90A: buffer_store_dwordx4 a[2:5], v0, s[12:15], s4 idxen offset:4095 ; encoding: [0xff,0x2f,0x7c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[2:5], v0, s[12:15], s4 idxen offset:4095

// GFX90A: buffer_store_dwordx4 a[2:5], v0, s[12:15], s4 offen offset:4095 ; encoding: [0xff,0x1f,0x7c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[2:5], v0, s[12:15], s4 offen offset:4095

// GFX90A: buffer_store_dwordx4 a[2:5], off, s[12:15], s4 ; encoding: [0x00,0x00,0x7c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[2:5], off, s[12:15], s4

// GFX90A: buffer_store_dwordx4 a[2:5], off, s[12:15], s4 ; encoding: [0x00,0x00,0x7c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[2:5], off, s[12:15], s4

// GFX90A: buffer_store_dwordx4 a[2:5], off, s[12:15], s4 offset:7 ; encoding: [0x07,0x00,0x7c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[2:5], off, s[12:15], s4 offset:7

// GFX90A: buffer_store_dwordx4 a[2:5], off, s[12:15], s4 offset:4095 glc ; encoding: [0xff,0x4f,0x7c,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[2:5], off, s[12:15], s4 offset:4095 glc

// GFX90A: buffer_store_dwordx4 a[2:5], off, s[12:15], s4 offset:4095 slc ; encoding: [0xff,0x0f,0x7e,0xe0,0x00,0x02,0x83,0x04]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_store_dwordx4 a[2:5], off, s[12:15], s4 offset:4095 slc

// GFX90A: buffer_load_ubyte_d16 a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_ubyte_d16 a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe0,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_ubyte_d16 a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe0,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_ubyte_d16 a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe0,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_ubyte_d16 a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe0,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_ubyte_d16 a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe0,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_ubyte_d16 a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe0,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_ubyte_d16 a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe0,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_ubyte_d16 a5, off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe0,0x00,0x05,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a5, off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_ubyte_d16 a5, off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe0,0x00,0x05,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a5, off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_ubyte_d16 a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x80,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_ubyte_d16 a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x80,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_ubyte_d16 a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x80,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a5, off, s[8:11], s3

// GFX90A: buffer_load_ubyte_d16 a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x80,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a5, off, s[8:11], s3

// GFX90A: buffer_load_ubyte_d16 a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x80,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_load_ubyte_d16 a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x80,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_ubyte_d16 a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x82,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16 a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_ubyte_d16_hi a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_ubyte_d16_hi a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe0,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_ubyte_d16_hi a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe0,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_ubyte_d16_hi a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe0,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_ubyte_d16_hi a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe0,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_ubyte_d16_hi a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe0,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_ubyte_d16_hi a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe0,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_ubyte_d16_hi a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe0,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_ubyte_d16_hi a5, off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe0,0x00,0x05,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a5, off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_ubyte_d16_hi a5, off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe0,0x00,0x05,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a5, off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_ubyte_d16_hi a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x84,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_ubyte_d16_hi a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x84,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_ubyte_d16_hi a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x84,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a5, off, s[8:11], s3

// GFX90A: buffer_load_ubyte_d16_hi a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x84,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a5, off, s[8:11], s3

// GFX90A: buffer_load_ubyte_d16_hi a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x84,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_load_ubyte_d16_hi a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x84,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_ubyte_d16_hi a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x86,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_ubyte_d16_hi a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_sbyte_d16 a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_sbyte_d16 a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe0,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_sbyte_d16 a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe0,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_sbyte_d16 a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe0,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_sbyte_d16 a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe0,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_sbyte_d16 a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe0,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_sbyte_d16 a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe0,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_sbyte_d16 a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe0,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_sbyte_d16 a5, off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe0,0x00,0x05,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a5, off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_sbyte_d16 a5, off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe0,0x00,0x05,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a5, off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_sbyte_d16 a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x88,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_sbyte_d16 a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x88,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_sbyte_d16 a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x88,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a5, off, s[8:11], s3

// GFX90A: buffer_load_sbyte_d16 a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x88,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a5, off, s[8:11], s3

// GFX90A: buffer_load_sbyte_d16 a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x88,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_load_sbyte_d16 a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x88,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_sbyte_d16 a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x8a,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16 a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_sbyte_d16_hi a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_sbyte_d16_hi a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe0,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_sbyte_d16_hi a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe0,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_sbyte_d16_hi a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe0,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_sbyte_d16_hi a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe0,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_sbyte_d16_hi a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe0,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_sbyte_d16_hi a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe0,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_sbyte_d16_hi a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe0,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_sbyte_d16_hi a5, off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe0,0x00,0x05,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a5, off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_sbyte_d16_hi a5, off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe0,0x00,0x05,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a5, off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_sbyte_d16_hi a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x8c,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_sbyte_d16_hi a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x8c,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_sbyte_d16_hi a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x8c,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a5, off, s[8:11], s3

// GFX90A: buffer_load_sbyte_d16_hi a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x8c,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a5, off, s[8:11], s3

// GFX90A: buffer_load_sbyte_d16_hi a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x8c,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_load_sbyte_d16_hi a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x8c,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_sbyte_d16_hi a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x8e,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_sbyte_d16_hi a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_short_d16 a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_short_d16 a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe0,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_short_d16 a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe0,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_short_d16 a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe0,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_short_d16 a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe0,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_short_d16 a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe0,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_short_d16 a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe0,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_short_d16 a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe0,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_short_d16 a5, off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe0,0x00,0x05,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a5, off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_short_d16 a5, off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe0,0x00,0x05,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a5, off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_short_d16 a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x90,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_short_d16 a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x90,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_short_d16 a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x90,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a5, off, s[8:11], s3

// GFX90A: buffer_load_short_d16 a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x90,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a5, off, s[8:11], s3

// GFX90A: buffer_load_short_d16 a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x90,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_load_short_d16 a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x90,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_short_d16 a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x92,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16 a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_load_short_d16_hi a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_short_d16_hi a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe0,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_load_short_d16_hi a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe0,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_load_short_d16_hi a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe0,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_load_short_d16_hi a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe0,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_load_short_d16_hi a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe0,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_load_short_d16_hi a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe0,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_load_short_d16_hi a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe0,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_load_short_d16_hi a5, off, s[8:11], 0.5 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe0,0x00,0x05,0x82,0xf0]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a5, off, s[8:11], 0.5 offset:4095

// GFX90A: buffer_load_short_d16_hi a5, off, s[8:11], -4.0 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe0,0x00,0x05,0x82,0xf7]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a5, off, s[8:11], -4.0 offset:4095

// GFX90A: buffer_load_short_d16_hi a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x94,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_load_short_d16_hi a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x94,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_load_short_d16_hi a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x94,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a5, off, s[8:11], s3

// GFX90A: buffer_load_short_d16_hi a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x94,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a5, off, s[8:11], s3

// GFX90A: buffer_load_short_d16_hi a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x94,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_load_short_d16_hi a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x94,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_load_short_d16_hi a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x96,0xe0,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_load_short_d16_hi a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_swap a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_swap a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe1,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_swap a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe1,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_swap a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe1,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_swap a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe1,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_swap a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe1,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_swap a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe1,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_swap a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x00,0xe1,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_swap a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x00,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_swap a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x00,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_swap a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x00,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap a5, off, s[8:11], s3

// GFX90A: buffer_atomic_swap a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x00,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap a5, off, s[8:11], s3

// GFX90A: buffer_atomic_swap a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x00,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_swap a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x00,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_swap a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x02,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_cmpswap a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_cmpswap a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe1,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_cmpswap a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe1,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_cmpswap a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe1,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_cmpswap a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe1,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_cmpswap a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe1,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_cmpswap a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe1,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_cmpswap a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x04,0xe1,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_cmpswap a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x04,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_cmpswap a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x04,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_cmpswap a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x04,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_cmpswap a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x04,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_cmpswap a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x04,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_cmpswap a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x04,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_cmpswap a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x06,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_add a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_add a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe1,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_add a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe1,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_add a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe1,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_add a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe1,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_add a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe1,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_add a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe1,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_add a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x08,0xe1,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_add a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x08,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_add a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x08,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_add a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x08,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add a5, off, s[8:11], s3

// GFX90A: buffer_atomic_add a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x08,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add a5, off, s[8:11], s3

// GFX90A: buffer_atomic_add a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x08,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_add a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x08,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_add a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x0a,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_sub a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_sub a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe1,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_sub a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe1,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_sub a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe1,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_sub a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe1,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_sub a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe1,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_sub a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe1,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_sub a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x0c,0xe1,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_sub a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x0c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_sub a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x0c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_sub a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x0c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub a5, off, s[8:11], s3

// GFX90A: buffer_atomic_sub a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x0c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub a5, off, s[8:11], s3

// GFX90A: buffer_atomic_sub a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x0c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_sub a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x0c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_sub a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x0e,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_smin a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_smin a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe1,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_smin a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe1,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_smin a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe1,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_smin a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe1,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_smin a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe1,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_smin a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe1,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_smin a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x10,0xe1,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_smin a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x10,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_smin a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x10,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_smin a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x10,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin a5, off, s[8:11], s3

// GFX90A: buffer_atomic_smin a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x10,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin a5, off, s[8:11], s3

// GFX90A: buffer_atomic_smin a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x10,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_smin a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x10,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_smin a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x12,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_umin a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_umin a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe1,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_umin a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe1,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_umin a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe1,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_umin a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe1,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_umin a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe1,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_umin a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe1,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_umin a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x14,0xe1,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_umin a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x14,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_umin a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x14,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_umin a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x14,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin a5, off, s[8:11], s3

// GFX90A: buffer_atomic_umin a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x14,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin a5, off, s[8:11], s3

// GFX90A: buffer_atomic_umin a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x14,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_umin a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x14,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_umin a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x16,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_smax a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_smax a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe1,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_smax a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe1,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_smax a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe1,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_smax a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe1,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_smax a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe1,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_smax a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe1,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_smax a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x18,0xe1,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_smax a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x18,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_smax a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x18,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_smax a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x18,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax a5, off, s[8:11], s3

// GFX90A: buffer_atomic_smax a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x18,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax a5, off, s[8:11], s3

// GFX90A: buffer_atomic_smax a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x18,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_smax a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x18,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_smax a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x1a,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_umax a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_umax a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe1,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_umax a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe1,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_umax a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe1,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_umax a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe1,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_umax a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe1,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_umax a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe1,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_umax a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x1c,0xe1,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_umax a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x1c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_umax a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x1c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_umax a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x1c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax a5, off, s[8:11], s3

// GFX90A: buffer_atomic_umax a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x1c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax a5, off, s[8:11], s3

// GFX90A: buffer_atomic_umax a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x1c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_umax a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x1c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_umax a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x1e,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_and a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_and a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe1,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_and a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe1,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_and a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe1,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_and a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe1,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_and a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe1,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_and a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe1,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_and a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe1,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_and a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x20,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_and a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x20,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_and a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x20,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and a5, off, s[8:11], s3

// GFX90A: buffer_atomic_and a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x20,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and a5, off, s[8:11], s3

// GFX90A: buffer_atomic_and a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x20,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_and a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x20,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_and a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x22,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_or a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_or a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe1,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_or a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe1,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_or a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe1,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_or a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe1,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_or a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe1,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_or a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe1,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_or a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x24,0xe1,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_or a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x24,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_or a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x24,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_or a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x24,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or a5, off, s[8:11], s3

// GFX90A: buffer_atomic_or a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x24,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or a5, off, s[8:11], s3

// GFX90A: buffer_atomic_or a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x24,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_or a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x24,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_or a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x26,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_xor a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_xor a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe1,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_xor a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe1,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_xor a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe1,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_xor a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe1,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_xor a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe1,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_xor a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe1,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_xor a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x28,0xe1,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_xor a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x28,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_xor a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x28,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_xor a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x28,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor a5, off, s[8:11], s3

// GFX90A: buffer_atomic_xor a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x28,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor a5, off, s[8:11], s3

// GFX90A: buffer_atomic_xor a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x28,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_xor a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x28,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_xor a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x2a,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_inc a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_inc a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe1,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_inc a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe1,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_inc a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe1,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_inc a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe1,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_inc a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe1,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_inc a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe1,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_inc a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x2c,0xe1,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_inc a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x2c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_inc a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x2c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_inc a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x2c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc a5, off, s[8:11], s3

// GFX90A: buffer_atomic_inc a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x2c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc a5, off, s[8:11], s3

// GFX90A: buffer_atomic_inc a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x2c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_inc a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x2c,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_inc a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x2e,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_dec a5, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec a5, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_dec a255, off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe1,0x00,0xff,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec a255, off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_dec a5, off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe1,0x00,0x05,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec a5, off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_dec a5, off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe1,0x00,0x05,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec a5, off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_dec a5, off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe1,0x00,0x05,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec a5, off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_dec a5, off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe1,0x00,0x05,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec a5, off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_dec a5, off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe1,0x00,0x05,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec a5, off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_dec a5, off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x30,0xe1,0x00,0x05,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec a5, off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_dec a5, v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x30,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec a5, v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_dec a5, v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x30,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec a5, v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_dec a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x30,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec a5, off, s[8:11], s3

// GFX90A: buffer_atomic_dec a5, off, s[8:11], s3 ; encoding: [0x00,0x00,0x30,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec a5, off, s[8:11], s3

// GFX90A: buffer_atomic_dec a5, off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x30,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec a5, off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_dec a5, off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x30,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec a5, off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_dec a5, off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x32,0xe1,0x00,0x05,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec a5, off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_swap_x2 a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap_x2 a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_swap_x2 a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe1,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap_x2 a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_swap_x2 a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe1,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap_x2 a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_swap_x2 a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe1,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap_x2 a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_swap_x2 a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe1,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap_x2 a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_swap_x2 a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe1,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap_x2 a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_swap_x2 a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe1,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap_x2 a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_swap_x2 a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x80,0xe1,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap_x2 a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_swap_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x80,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_swap_x2 a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x80,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap_x2 a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_swap_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x80,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_swap_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x80,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_swap_x2 a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x80,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap_x2 a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_swap_x2 a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x80,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap_x2 a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_swap_x2 a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x82,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_swap_x2 a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_cmpswap_x2 a[252:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe1,0x00,0xfc,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap_x2 a[252:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_cmpswap_x2 a[6:9], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe1,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap_x2 a[6:9], off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_cmpswap_x2 a[6:9], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe1,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap_x2 a[6:9], off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe1,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe1,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe1,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x84,0xe1,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_cmpswap_x2 a[6:9], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x84,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap_x2 a[6:9], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_cmpswap_x2 a[6:9], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x84,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap_x2 a[6:9], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], s3 ; encoding: [0x00,0x00,0x84,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], s3

// GFX90A: buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], s3 ; encoding: [0x00,0x00,0x84,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], s3

// GFX90A: buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x84,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x84,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x86,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_cmpswap_x2 a[6:9], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_add_x2 a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add_x2 a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_add_x2 a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe1,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add_x2 a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_add_x2 a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe1,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add_x2 a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_add_x2 a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe1,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add_x2 a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_add_x2 a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe1,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add_x2 a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_add_x2 a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe1,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add_x2 a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_add_x2 a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe1,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add_x2 a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_add_x2 a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x88,0xe1,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add_x2 a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_add_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x88,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_add_x2 a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x88,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add_x2 a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_add_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x88,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_add_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x88,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_add_x2 a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x88,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add_x2 a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_add_x2 a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x88,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add_x2 a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_add_x2 a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x8a,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_add_x2 a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_sub_x2 a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub_x2 a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_sub_x2 a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe1,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub_x2 a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_sub_x2 a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe1,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub_x2 a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_sub_x2 a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe1,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub_x2 a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_sub_x2 a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe1,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub_x2 a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_sub_x2 a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe1,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub_x2 a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_sub_x2 a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe1,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub_x2 a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_sub_x2 a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x8c,0xe1,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub_x2 a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_sub_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x8c,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_sub_x2 a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x8c,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub_x2 a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_sub_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x8c,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_sub_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x8c,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_sub_x2 a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x8c,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub_x2 a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_sub_x2 a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x8c,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub_x2 a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_sub_x2 a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x8e,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_sub_x2 a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_smin_x2 a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin_x2 a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_smin_x2 a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe1,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin_x2 a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_smin_x2 a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe1,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin_x2 a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_smin_x2 a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe1,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin_x2 a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_smin_x2 a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe1,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin_x2 a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_smin_x2 a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe1,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin_x2 a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_smin_x2 a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe1,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin_x2 a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_smin_x2 a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x90,0xe1,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin_x2 a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_smin_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x90,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_smin_x2 a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x90,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin_x2 a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_smin_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x90,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_smin_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x90,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_smin_x2 a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x90,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin_x2 a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_smin_x2 a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x90,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin_x2 a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_smin_x2 a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x92,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smin_x2 a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_umin_x2 a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin_x2 a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_umin_x2 a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe1,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin_x2 a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_umin_x2 a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe1,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin_x2 a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_umin_x2 a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe1,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin_x2 a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_umin_x2 a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe1,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin_x2 a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_umin_x2 a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe1,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin_x2 a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_umin_x2 a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe1,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin_x2 a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_umin_x2 a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x94,0xe1,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin_x2 a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_umin_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x94,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_umin_x2 a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x94,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin_x2 a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_umin_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x94,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_umin_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x94,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_umin_x2 a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x94,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin_x2 a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_umin_x2 a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x94,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin_x2 a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_umin_x2 a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x96,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umin_x2 a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_smax_x2 a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x98,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax_x2 a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_smax_x2 a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x98,0xe1,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax_x2 a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_smax_x2 a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x98,0xe1,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax_x2 a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_smax_x2 a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x98,0xe1,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax_x2 a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_smax_x2 a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x98,0xe1,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax_x2 a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_smax_x2 a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x98,0xe1,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax_x2 a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_smax_x2 a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x98,0xe1,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax_x2 a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_smax_x2 a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x98,0xe1,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax_x2 a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_smax_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x98,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_smax_x2 a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x98,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax_x2 a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_smax_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x98,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_smax_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x98,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_smax_x2 a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x98,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax_x2 a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_smax_x2 a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x98,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax_x2 a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_smax_x2 a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x9a,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_smax_x2 a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_umax_x2 a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x9c,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax_x2 a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_umax_x2 a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0x9c,0xe1,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax_x2 a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_umax_x2 a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0x9c,0xe1,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax_x2 a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_umax_x2 a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0x9c,0xe1,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax_x2 a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_umax_x2 a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0x9c,0xe1,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax_x2 a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_umax_x2 a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0x9c,0xe1,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax_x2 a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_umax_x2 a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0x9c,0xe1,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax_x2 a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_umax_x2 a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0x9c,0xe1,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax_x2 a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_umax_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0x9c,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_umax_x2 a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0x9c,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax_x2 a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_umax_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x9c,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_umax_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0x9c,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_umax_x2 a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0x9c,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax_x2 a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_umax_x2 a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0x9c,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax_x2 a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_umax_x2 a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0x9e,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_umax_x2 a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_and_x2 a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xa0,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and_x2 a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_and_x2 a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xa0,0xe1,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and_x2 a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_and_x2 a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0xa0,0xe1,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and_x2 a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_and_x2 a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0xa0,0xe1,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and_x2 a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_and_x2 a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0xa0,0xe1,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and_x2 a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_and_x2 a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0xa0,0xe1,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and_x2 a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_and_x2 a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0xa0,0xe1,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and_x2 a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_and_x2 a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0xa0,0xe1,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and_x2 a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_and_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0xa0,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_and_x2 a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0xa0,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and_x2 a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_and_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0xa0,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_and_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0xa0,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_and_x2 a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0xa0,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and_x2 a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_and_x2 a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0xa0,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and_x2 a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_and_x2 a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0xa2,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_and_x2 a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_or_x2 a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xa4,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or_x2 a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_or_x2 a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xa4,0xe1,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or_x2 a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_or_x2 a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0xa4,0xe1,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or_x2 a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_or_x2 a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0xa4,0xe1,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or_x2 a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_or_x2 a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0xa4,0xe1,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or_x2 a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_or_x2 a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0xa4,0xe1,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or_x2 a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_or_x2 a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0xa4,0xe1,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or_x2 a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_or_x2 a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0xa4,0xe1,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or_x2 a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_or_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0xa4,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_or_x2 a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0xa4,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or_x2 a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_or_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0xa4,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_or_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0xa4,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_or_x2 a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0xa4,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or_x2 a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_or_x2 a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0xa4,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or_x2 a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_or_x2 a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0xa6,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_or_x2 a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_xor_x2 a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xa8,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor_x2 a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_xor_x2 a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xa8,0xe1,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor_x2 a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_xor_x2 a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0xa8,0xe1,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor_x2 a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_xor_x2 a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0xa8,0xe1,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor_x2 a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_xor_x2 a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0xa8,0xe1,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor_x2 a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_xor_x2 a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0xa8,0xe1,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor_x2 a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_xor_x2 a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0xa8,0xe1,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor_x2 a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_xor_x2 a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0xa8,0xe1,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor_x2 a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_xor_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0xa8,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_xor_x2 a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0xa8,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor_x2 a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_xor_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0xa8,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_xor_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0xa8,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_xor_x2 a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0xa8,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor_x2 a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_xor_x2 a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0xa8,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor_x2 a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_xor_x2 a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0xaa,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_xor_x2 a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_inc_x2 a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xac,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc_x2 a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_inc_x2 a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xac,0xe1,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc_x2 a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_inc_x2 a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0xac,0xe1,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc_x2 a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_inc_x2 a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0xac,0xe1,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc_x2 a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_inc_x2 a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0xac,0xe1,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc_x2 a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_inc_x2 a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0xac,0xe1,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc_x2 a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_inc_x2 a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0xac,0xe1,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc_x2 a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_inc_x2 a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0xac,0xe1,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc_x2 a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_inc_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0xac,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_inc_x2 a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0xac,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc_x2 a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_inc_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0xac,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_inc_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0xac,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_inc_x2 a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0xac,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc_x2 a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_inc_x2 a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0xac,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc_x2 a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_inc_x2 a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0xae,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_inc_x2 a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: buffer_atomic_dec_x2 a[6:7], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xb0,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec_x2 a[6:7], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_dec_x2 a[254:255], off, s[8:11], s3 offset:4095 ; encoding: [0xff,0x0f,0xb0,0xe1,0x00,0xfe,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec_x2 a[254:255], off, s[8:11], s3 offset:4095

// GFX90A: buffer_atomic_dec_x2 a[6:7], off, s[12:15], s3 offset:4095 ; encoding: [0xff,0x0f,0xb0,0xe1,0x00,0x06,0x83,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec_x2 a[6:7], off, s[12:15], s3 offset:4095

// GFX90A: buffer_atomic_dec_x2 a[6:7], off, s[96:99], s3 offset:4095 ; encoding: [0xff,0x0f,0xb0,0xe1,0x00,0x06,0x98,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec_x2 a[6:7], off, s[96:99], s3 offset:4095

// GFX90A: buffer_atomic_dec_x2 a[6:7], off, s[8:11], s101 offset:4095 ; encoding: [0xff,0x0f,0xb0,0xe1,0x00,0x06,0x82,0x65]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec_x2 a[6:7], off, s[8:11], s101 offset:4095

// GFX90A: buffer_atomic_dec_x2 a[6:7], off, s[8:11], m0 offset:4095 ; encoding: [0xff,0x0f,0xb0,0xe1,0x00,0x06,0x82,0x7c]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec_x2 a[6:7], off, s[8:11], m0 offset:4095

// GFX90A: buffer_atomic_dec_x2 a[6:7], off, s[8:11], 0 offset:4095 ; encoding: [0xff,0x0f,0xb0,0xe1,0x00,0x06,0x82,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec_x2 a[6:7], off, s[8:11], 0 offset:4095

// GFX90A: buffer_atomic_dec_x2 a[6:7], off, s[8:11], -1 offset:4095 ; encoding: [0xff,0x0f,0xb0,0xe1,0x00,0x06,0x82,0xc1]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec_x2 a[6:7], off, s[8:11], -1 offset:4095

// GFX90A: buffer_atomic_dec_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095 ; encoding: [0xff,0x2f,0xb0,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec_x2 a[6:7], v0, s[8:11], s3 idxen offset:4095

// GFX90A: buffer_atomic_dec_x2 a[6:7], v0, s[8:11], s3 offen offset:4095 ; encoding: [0xff,0x1f,0xb0,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec_x2 a[6:7], v0, s[8:11], s3 offen offset:4095

// GFX90A: buffer_atomic_dec_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0xb0,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_dec_x2 a[6:7], off, s[8:11], s3 ; encoding: [0x00,0x00,0xb0,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec_x2 a[6:7], off, s[8:11], s3

// GFX90A: buffer_atomic_dec_x2 a[6:7], off, s[8:11], s3 offset:7 ; encoding: [0x07,0x00,0xb0,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec_x2 a[6:7], off, s[8:11], s3 offset:7

// GFX90A: buffer_atomic_dec_x2 a[6:7], off, s[8:11], s3 offset:4095 glc ; encoding: [0xff,0x4f,0xb0,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec_x2 a[6:7], off, s[8:11], s3 offset:4095 glc

// GFX90A: buffer_atomic_dec_x2 a[6:7], off, s[8:11], s3 offset:4095 slc ; encoding: [0xff,0x0f,0xb2,0xe1,0x00,0x06,0x82,0x03]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
buffer_atomic_dec_x2 a[6:7], off, s[8:11], s3 offset:4095 slc

// GFX90A: tbuffer_load_format_x a1, off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x78,0xe9,0x00,0x01,0x81,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
tbuffer_load_format_x a1, off, s[4:7],  dfmt:15, nfmt:2, s1

// GFX90A: tbuffer_load_format_xy a[2:3], off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x80,0x78,0xe9,0x00,0x02,0x81,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
tbuffer_load_format_xy a[2:3], off, s[4:7],  dfmt:15, nfmt:2, s1

// GFX90A: tbuffer_load_format_xyz a[2:4], off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x79,0xe9,0x00,0x02,0x81,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
tbuffer_load_format_xyz a[2:4], off, s[4:7],  dfmt:15, nfmt:2, s1

// GFX90A: tbuffer_load_format_xyzw a[2:5], off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x80,0x79,0xe9,0x00,0x02,0x81,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
tbuffer_load_format_xyzw a[2:5], off, s[4:7],  dfmt:15, nfmt:2, s1

// GFX90A: tbuffer_store_format_x a1, off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x00,0x7a,0xe9,0x00,0x01,0x81,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
tbuffer_store_format_x a1, off, s[4:7],  dfmt:15, nfmt:2, s1

// GFX90A: tbuffer_store_format_xy a[2:3], off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x80,0x7a,0xe9,0x00,0x02,0x81,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
tbuffer_store_format_xy a[2:3], off, s[4:7],  dfmt:15, nfmt:2, s1

// GFX90A: tbuffer_store_format_xyzw a[2:5], off, s[4:7], s1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x80,0x7b,0xe9,0x00,0x02,0x81,0x01]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
tbuffer_store_format_xyzw a[2:5], off, s[4:7],  dfmt:15, nfmt:2, s1

// GFX90A: tbuffer_store_format_xyzw a[2:5], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x80,0x7b,0xe9,0x00,0x02,0x9c,0x6d]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
tbuffer_store_format_xyzw a[2:5], off, ttmp[4:7],  dfmt:15, nfmt:2, ttmp1

// GFX90A: tbuffer_store_format_xyzw a[2:5], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_RESERVED_15] ; encoding: [0x00,0x80,0x7b,0xe8,0x00,0x02,0x9c,0x6d]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
tbuffer_store_format_xyzw a[2:5], off, ttmp[4:7],  dfmt:15, nfmt:0, ttmp1

// GFX90A: tbuffer_store_format_xyzw a[2:5], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_INVALID,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x80,0x03,0xe9,0x00,0x02,0x9c,0x6d]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
tbuffer_store_format_xyzw a[2:5], off, ttmp[4:7],  dfmt:0, nfmt:2, ttmp1

// GFX90A: tbuffer_store_format_xyzw a[2:5], off, ttmp[4:7], ttmp1 format:[BUF_DATA_FORMAT_RESERVED_15,BUF_NUM_FORMAT_USCALED] ; encoding: [0x00,0x80,0x7b,0xe9,0x00,0x02,0x9c,0x6d]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
tbuffer_store_format_xyzw a[2:5], off, ttmp[4:7],  dfmt:15, nfmt:2, ttmp1

// GFX90A: ds_add_u32 v1, a2 offset:65535  ; encoding: [0xff,0xff,0x00,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_u32 v1, a2 offset:65535

// GFX90A: ds_add_u32 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x00,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_u32 v255, a2 offset:65535

// GFX90A: ds_add_u32 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x00,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_u32 v1, a255 offset:65535

// GFX90A: ds_add_u32 v1, a2               ; encoding: [0x00,0x00,0x00,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_u32 v1, a2

// GFX90A: ds_add_u32 v1, a2               ; encoding: [0x00,0x00,0x00,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_u32 v1, a2

// GFX90A: ds_add_u32 v1, a2 offset:4      ; encoding: [0x04,0x00,0x00,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_u32 v1, a2 offset:4

// GFX90A: ds_sub_u32 v1, a2 offset:65535  ; encoding: [0xff,0xff,0x02,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_sub_u32 v1, a2 offset:65535

// GFX90A: ds_sub_u32 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x02,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_sub_u32 v255, a2 offset:65535

// GFX90A: ds_sub_u32 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x02,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_sub_u32 v1, a255 offset:65535

// GFX90A: ds_sub_u32 v1, a2               ; encoding: [0x00,0x00,0x02,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_sub_u32 v1, a2

// GFX90A: ds_sub_u32 v1, a2               ; encoding: [0x00,0x00,0x02,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_sub_u32 v1, a2

// GFX90A: ds_sub_u32 v1, a2 offset:4      ; encoding: [0x04,0x00,0x02,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_sub_u32 v1, a2 offset:4

// GFX90A: ds_rsub_u32 v1, a2 offset:65535 ; encoding: [0xff,0xff,0x04,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_rsub_u32 v1, a2 offset:65535

// GFX90A: ds_rsub_u32 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x04,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_rsub_u32 v255, a2 offset:65535

// GFX90A: ds_rsub_u32 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x04,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_rsub_u32 v1, a255 offset:65535

// GFX90A: ds_rsub_u32 v1, a2              ; encoding: [0x00,0x00,0x04,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_rsub_u32 v1, a2

// GFX90A: ds_rsub_u32 v1, a2              ; encoding: [0x00,0x00,0x04,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_rsub_u32 v1, a2

// GFX90A: ds_rsub_u32 v1, a2 offset:4     ; encoding: [0x04,0x00,0x04,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_rsub_u32 v1, a2 offset:4

// GFX90A: ds_inc_u32 v1, a2 offset:65535  ; encoding: [0xff,0xff,0x06,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_inc_u32 v1, a2 offset:65535

// GFX90A: ds_inc_u32 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x06,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_inc_u32 v255, a2 offset:65535

// GFX90A: ds_inc_u32 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x06,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_inc_u32 v1, a255 offset:65535

// GFX90A: ds_inc_u32 v1, a2               ; encoding: [0x00,0x00,0x06,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_inc_u32 v1, a2

// GFX90A: ds_inc_u32 v1, a2               ; encoding: [0x00,0x00,0x06,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_inc_u32 v1, a2

// GFX90A: ds_inc_u32 v1, a2 offset:4      ; encoding: [0x04,0x00,0x06,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_inc_u32 v1, a2 offset:4

// GFX90A: ds_dec_u32 v1, a2 offset:65535  ; encoding: [0xff,0xff,0x08,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_dec_u32 v1, a2 offset:65535

// GFX90A: ds_dec_u32 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x08,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_dec_u32 v255, a2 offset:65535

// GFX90A: ds_dec_u32 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x08,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_dec_u32 v1, a255 offset:65535

// GFX90A: ds_dec_u32 v1, a2               ; encoding: [0x00,0x00,0x08,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_dec_u32 v1, a2

// GFX90A: ds_dec_u32 v1, a2               ; encoding: [0x00,0x00,0x08,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_dec_u32 v1, a2

// GFX90A: ds_dec_u32 v1, a2 offset:4      ; encoding: [0x04,0x00,0x08,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_dec_u32 v1, a2 offset:4

// GFX90A: ds_min_i32 v1, a2 offset:65535  ; encoding: [0xff,0xff,0x0a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_i32 v1, a2 offset:65535

// GFX90A: ds_min_i32 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x0a,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_i32 v255, a2 offset:65535

// GFX90A: ds_min_i32 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x0a,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_i32 v1, a255 offset:65535

// GFX90A: ds_min_i32 v1, a2               ; encoding: [0x00,0x00,0x0a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_i32 v1, a2

// GFX90A: ds_min_i32 v1, a2               ; encoding: [0x00,0x00,0x0a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_i32 v1, a2

// GFX90A: ds_min_i32 v1, a2 offset:4      ; encoding: [0x04,0x00,0x0a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_i32 v1, a2 offset:4

// GFX90A: ds_max_i32 v1, a2 offset:65535  ; encoding: [0xff,0xff,0x0c,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_i32 v1, a2 offset:65535

// GFX90A: ds_max_i32 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x0c,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_i32 v255, a2 offset:65535

// GFX90A: ds_max_i32 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x0c,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_i32 v1, a255 offset:65535

// GFX90A: ds_max_i32 v1, a2               ; encoding: [0x00,0x00,0x0c,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_i32 v1, a2

// GFX90A: ds_max_i32 v1, a2               ; encoding: [0x00,0x00,0x0c,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_i32 v1, a2

// GFX90A: ds_max_i32 v1, a2 offset:4      ; encoding: [0x04,0x00,0x0c,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_i32 v1, a2 offset:4

// GFX90A: ds_min_u32 v1, a2 offset:65535  ; encoding: [0xff,0xff,0x0e,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_u32 v1, a2 offset:65535

// GFX90A: ds_min_u32 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x0e,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_u32 v255, a2 offset:65535

// GFX90A: ds_min_u32 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x0e,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_u32 v1, a255 offset:65535

// GFX90A: ds_min_u32 v1, a2               ; encoding: [0x00,0x00,0x0e,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_u32 v1, a2

// GFX90A: ds_min_u32 v1, a2               ; encoding: [0x00,0x00,0x0e,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_u32 v1, a2

// GFX90A: ds_min_u32 v1, a2 offset:4      ; encoding: [0x04,0x00,0x0e,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_u32 v1, a2 offset:4

// GFX90A: ds_max_u32 v1, a2 offset:65535  ; encoding: [0xff,0xff,0x10,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_u32 v1, a2 offset:65535

// GFX90A: ds_max_u32 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x10,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_u32 v255, a2 offset:65535

// GFX90A: ds_max_u32 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x10,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_u32 v1, a255 offset:65535

// GFX90A: ds_max_u32 v1, a2               ; encoding: [0x00,0x00,0x10,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_u32 v1, a2

// GFX90A: ds_max_u32 v1, a2               ; encoding: [0x00,0x00,0x10,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_u32 v1, a2

// GFX90A: ds_max_u32 v1, a2 offset:4      ; encoding: [0x04,0x00,0x10,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_u32 v1, a2 offset:4

// GFX90A: ds_and_b32 v1, a2 offset:65535  ; encoding: [0xff,0xff,0x12,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_and_b32 v1, a2 offset:65535

// GFX90A: ds_and_b32 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x12,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_and_b32 v255, a2 offset:65535

// GFX90A: ds_and_b32 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x12,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_and_b32 v1, a255 offset:65535

// GFX90A: ds_and_b32 v1, a2               ; encoding: [0x00,0x00,0x12,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_and_b32 v1, a2

// GFX90A: ds_and_b32 v1, a2               ; encoding: [0x00,0x00,0x12,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_and_b32 v1, a2

// GFX90A: ds_and_b32 v1, a2 offset:4      ; encoding: [0x04,0x00,0x12,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_and_b32 v1, a2 offset:4

// GFX90A: ds_or_b32 v1, a2 offset:65535   ; encoding: [0xff,0xff,0x14,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_or_b32 v1, a2 offset:65535

// GFX90A: ds_or_b32 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x14,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_or_b32 v255, a2 offset:65535

// GFX90A: ds_or_b32 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x14,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_or_b32 v1, a255 offset:65535

// GFX90A: ds_or_b32 v1, a2                ; encoding: [0x00,0x00,0x14,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_or_b32 v1, a2

// GFX90A: ds_or_b32 v1, a2                ; encoding: [0x00,0x00,0x14,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_or_b32 v1, a2

// GFX90A: ds_or_b32 v1, a2 offset:4       ; encoding: [0x04,0x00,0x14,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_or_b32 v1, a2 offset:4

// GFX90A: ds_xor_b32 v1, a2 offset:65535  ; encoding: [0xff,0xff,0x16,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_xor_b32 v1, a2 offset:65535

// GFX90A: ds_xor_b32 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x16,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_xor_b32 v255, a2 offset:65535

// GFX90A: ds_xor_b32 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x16,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_xor_b32 v1, a255 offset:65535

// GFX90A: ds_xor_b32 v1, a2               ; encoding: [0x00,0x00,0x16,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_xor_b32 v1, a2

// GFX90A: ds_xor_b32 v1, a2               ; encoding: [0x00,0x00,0x16,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_xor_b32 v1, a2

// GFX90A: ds_xor_b32 v1, a2 offset:4      ; encoding: [0x04,0x00,0x16,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_xor_b32 v1, a2 offset:4

// GFX90A: ds_mskor_b32 v1, a2, a3 offset:65535 ; encoding: [0xff,0xff,0x18,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_b32 v1, a2, a3 offset:65535

// GFX90A: ds_mskor_b32 v255, a2, a3 offset:65535 ; encoding: [0xff,0xff,0x18,0xda,0xff,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_b32 v255, a2, a3 offset:65535

// GFX90A: ds_mskor_b32 v1, a255, a3 offset:65535 ; encoding: [0xff,0xff,0x18,0xda,0x01,0xff,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_b32 v1, a255, a3 offset:65535

// GFX90A: ds_mskor_b32 v1, a2, a255 offset:65535 ; encoding: [0xff,0xff,0x18,0xda,0x01,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_b32 v1, a2, a255 offset:65535

// GFX90A: ds_mskor_b32 v1, a2, a3         ; encoding: [0x00,0x00,0x18,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_b32 v1, a2, a3

// GFX90A: ds_mskor_b32 v1, a2, a3         ; encoding: [0x00,0x00,0x18,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_b32 v1, a2, a3

// GFX90A: ds_mskor_b32 v1, a2, a3 offset:4 ; encoding: [0x04,0x00,0x18,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_b32 v1, a2, a3 offset:4

// GFX90A: ds_write_b32 v1, a2 offset:65535 ; encoding: [0xff,0xff,0x1a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b32 v1, a2 offset:65535

// GFX90A: ds_write_b32 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x1a,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b32 v255, a2 offset:65535

// GFX90A: ds_write_b32 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x1a,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b32 v1, a255 offset:65535

// GFX90A: ds_write_b32 v1, a2             ; encoding: [0x00,0x00,0x1a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b32 v1, a2

// GFX90A: ds_write_b32 v1, a2             ; encoding: [0x00,0x00,0x1a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b32 v1, a2

// GFX90A: ds_write_b32 v1, a2 offset:4    ; encoding: [0x04,0x00,0x1a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b32 v1, a2 offset:4

// GFX90A: ds_write2_b32 v1, a2, a3 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x1c,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b32 v1, a2, a3 offset0:127 offset1:255

// GFX90A: ds_write2_b32 v255, a2, a3 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x1c,0xda,0xff,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b32 v255, a2, a3 offset0:127 offset1:255

// GFX90A: ds_write2_b32 v1, a255, a3 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x1c,0xda,0x01,0xff,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b32 v1, a255, a3 offset0:127 offset1:255

// GFX90A: ds_write2_b32 v1, a2, a255 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x1c,0xda,0x01,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b32 v1, a2, a255 offset0:127 offset1:255

// GFX90A: ds_write2_b32 v1, a2, a3 offset1:255 ; encoding: [0x00,0xff,0x1c,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b32 v1, a2, a3 offset1:255

// GFX90A: ds_write2_b32 v1, a2, a3 offset1:255 ; encoding: [0x00,0xff,0x1c,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b32 v1, a2, a3 offset1:255

// GFX90A: ds_write2_b32 v1, a2, a3 offset0:16 offset1:255 ; encoding: [0x10,0xff,0x1c,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b32 v1, a2, a3 offset0:16 offset1:255

// GFX90A: ds_write2_b32 v1, a2, a3 offset0:127 ; encoding: [0x7f,0x00,0x1c,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b32 v1, a2, a3 offset0:127

// GFX90A: ds_write2_b32 v1, a2, a3 offset0:127 ; encoding: [0x7f,0x00,0x1c,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b32 v1, a2, a3 offset0:127

// GFX90A: ds_write2_b32 v1, a2, a3 offset0:127 offset1:1 ; encoding: [0x7f,0x01,0x1c,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b32 v1, a2, a3 offset0:127 offset1:1

// GFX90A: ds_write2st64_b32 v1, a2, a3 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x1e,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b32 v1, a2, a3 offset0:127 offset1:255

// GFX90A: ds_write2st64_b32 v255, a2, a3 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x1e,0xda,0xff,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b32 v255, a2, a3 offset0:127 offset1:255

// GFX90A: ds_write2st64_b32 v1, a255, a3 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x1e,0xda,0x01,0xff,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b32 v1, a255, a3 offset0:127 offset1:255

// GFX90A: ds_write2st64_b32 v1, a2, a255 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x1e,0xda,0x01,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b32 v1, a2, a255 offset0:127 offset1:255

// GFX90A: ds_write2st64_b32 v1, a2, a3 offset1:255 ; encoding: [0x00,0xff,0x1e,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b32 v1, a2, a3 offset1:255

// GFX90A: ds_write2st64_b32 v1, a2, a3 offset1:255 ; encoding: [0x00,0xff,0x1e,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b32 v1, a2, a3 offset1:255

// GFX90A: ds_write2st64_b32 v1, a2, a3 offset0:16 offset1:255 ; encoding: [0x10,0xff,0x1e,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b32 v1, a2, a3 offset0:16 offset1:255

// GFX90A: ds_write2st64_b32 v1, a2, a3 offset0:127 ; encoding: [0x7f,0x00,0x1e,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b32 v1, a2, a3 offset0:127

// GFX90A: ds_write2st64_b32 v1, a2, a3 offset0:127 ; encoding: [0x7f,0x00,0x1e,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b32 v1, a2, a3 offset0:127

// GFX90A: ds_write2st64_b32 v1, a2, a3 offset0:127 offset1:1 ; encoding: [0x7f,0x01,0x1e,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b32 v1, a2, a3 offset0:127 offset1:1

// GFX90A: ds_cmpst_b32 v1, a2, a3 offset:65535 ; encoding: [0xff,0xff,0x20,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_b32 v1, a2, a3 offset:65535

// GFX90A: ds_cmpst_b32 v255, a2, a3 offset:65535 ; encoding: [0xff,0xff,0x20,0xda,0xff,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_b32 v255, a2, a3 offset:65535

// GFX90A: ds_cmpst_b32 v1, a255, a3 offset:65535 ; encoding: [0xff,0xff,0x20,0xda,0x01,0xff,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_b32 v1, a255, a3 offset:65535

// GFX90A: ds_cmpst_b32 v1, a2, a255 offset:65535 ; encoding: [0xff,0xff,0x20,0xda,0x01,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_b32 v1, a2, a255 offset:65535

// GFX90A: ds_cmpst_b32 v1, a2, a3         ; encoding: [0x00,0x00,0x20,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_b32 v1, a2, a3

// GFX90A: ds_cmpst_b32 v1, a2, a3         ; encoding: [0x00,0x00,0x20,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_b32 v1, a2, a3

// GFX90A: ds_cmpst_b32 v1, a2, a3 offset:4 ; encoding: [0x04,0x00,0x20,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_b32 v1, a2, a3 offset:4

// GFX90A: ds_cmpst_f32 v1, a2, a3 offset:65535 ; encoding: [0xff,0xff,0x22,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_f32 v1, a2, a3 offset:65535

// GFX90A: ds_cmpst_f32 v255, a2, a3 offset:65535 ; encoding: [0xff,0xff,0x22,0xda,0xff,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_f32 v255, a2, a3 offset:65535

// GFX90A: ds_cmpst_f32 v1, a255, a3 offset:65535 ; encoding: [0xff,0xff,0x22,0xda,0x01,0xff,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_f32 v1, a255, a3 offset:65535

// GFX90A: ds_cmpst_f32 v1, a2, a255 offset:65535 ; encoding: [0xff,0xff,0x22,0xda,0x01,0x02,0xff,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_f32 v1, a2, a255 offset:65535

// GFX90A: ds_cmpst_f32 v1, a2, a3         ; encoding: [0x00,0x00,0x22,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_f32 v1, a2, a3

// GFX90A: ds_cmpst_f32 v1, a2, a3         ; encoding: [0x00,0x00,0x22,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_f32 v1, a2, a3

// GFX90A: ds_cmpst_f32 v1, a2, a3 offset:4 ; encoding: [0x04,0x00,0x22,0xda,0x01,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_f32 v1, a2, a3 offset:4

// GFX90A: ds_min_f32 v1, a2 offset:65535  ; encoding: [0xff,0xff,0x24,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_f32 v1, a2 offset:65535

// GFX90A: ds_min_f32 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x24,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_f32 v255, a2 offset:65535

// GFX90A: ds_min_f32 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x24,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_f32 v1, a255 offset:65535

// GFX90A: ds_min_f32 v1, a2               ; encoding: [0x00,0x00,0x24,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_f32 v1, a2

// GFX90A: ds_min_f32 v1, a2               ; encoding: [0x00,0x00,0x24,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_f32 v1, a2

// GFX90A: ds_min_f32 v1, a2 offset:4      ; encoding: [0x04,0x00,0x24,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_f32 v1, a2 offset:4

// GFX90A: ds_max_f32 v1, a2 offset:65535  ; encoding: [0xff,0xff,0x26,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_f32 v1, a2 offset:65535

// GFX90A: ds_max_f32 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x26,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_f32 v255, a2 offset:65535

// GFX90A: ds_max_f32 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x26,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_f32 v1, a255 offset:65535

// GFX90A: ds_max_f32 v1, a2               ; encoding: [0x00,0x00,0x26,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_f32 v1, a2

// GFX90A: ds_max_f32 v1, a2               ; encoding: [0x00,0x00,0x26,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_f32 v1, a2

// GFX90A: ds_max_f32 v1, a2 offset:4      ; encoding: [0x04,0x00,0x26,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_f32 v1, a2 offset:4

// GFX90A: ds_add_f32 v1, a2 offset:65535  ; encoding: [0xff,0xff,0x2a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_f32 v1, a2 offset:65535

// GFX90A: ds_add_f32 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x2a,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_f32 v255, a2 offset:65535

// GFX90A: ds_add_f32 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x2a,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_f32 v1, a255 offset:65535

// GFX90A: ds_add_f32 v1, a2               ; encoding: [0x00,0x00,0x2a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_f32 v1, a2

// GFX90A: ds_add_f32 v1, a2               ; encoding: [0x00,0x00,0x2a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_f32 v1, a2

// GFX90A: ds_add_f32 v1, a2 offset:4      ; encoding: [0x04,0x00,0x2a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_f32 v1, a2 offset:4

// GFX90A: ds_write_b8 v1, a2 offset:65535 ; encoding: [0xff,0xff,0x3c,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b8 v1, a2 offset:65535

// GFX90A: ds_write_b8 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x3c,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b8 v255, a2 offset:65535

// GFX90A: ds_write_b8 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x3c,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b8 v1, a255 offset:65535

// GFX90A: ds_write_b8 v1, a2              ; encoding: [0x00,0x00,0x3c,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b8 v1, a2

// GFX90A: ds_write_b8 v1, a2              ; encoding: [0x00,0x00,0x3c,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b8 v1, a2

// GFX90A: ds_write_b8 v1, a2 offset:4     ; encoding: [0x04,0x00,0x3c,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b8 v1, a2 offset:4

// GFX90A: ds_write_b16 v1, a2 offset:65535 ; encoding: [0xff,0xff,0x3e,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b16 v1, a2 offset:65535

// GFX90A: ds_write_b16 v255, a2 offset:65535 ; encoding: [0xff,0xff,0x3e,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b16 v255, a2 offset:65535

// GFX90A: ds_write_b16 v1, a255 offset:65535 ; encoding: [0xff,0xff,0x3e,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b16 v1, a255 offset:65535

// GFX90A: ds_write_b16 v1, a2             ; encoding: [0x00,0x00,0x3e,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b16 v1, a2

// GFX90A: ds_write_b16 v1, a2             ; encoding: [0x00,0x00,0x3e,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b16 v1, a2

// GFX90A: ds_write_b16 v1, a2 offset:4    ; encoding: [0x04,0x00,0x3e,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b16 v1, a2 offset:4

// GFX90A: ds_add_rtn_u32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x40,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_u32 a5, v1, a2 offset:65535

// GFX90A: ds_add_rtn_u32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x40,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_u32 a255, v1, a2 offset:65535

// GFX90A: ds_add_rtn_u32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x40,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_u32 a5, v255, a2 offset:65535

// GFX90A: ds_add_rtn_u32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x40,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_u32 a5, v1, a255 offset:65535

// GFX90A: ds_add_rtn_u32 a5, v1, a2       ; encoding: [0x00,0x00,0x40,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_u32 a5, v1, a2

// GFX90A: ds_add_rtn_u32 a5, v1, a2       ; encoding: [0x00,0x00,0x40,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_u32 a5, v1, a2

// GFX90A: ds_add_rtn_u32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x40,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_u32 a5, v1, a2 offset:4

// GFX90A: ds_sub_rtn_u32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x42,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_sub_rtn_u32 a5, v1, a2 offset:65535

// GFX90A: ds_sub_rtn_u32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x42,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_sub_rtn_u32 a255, v1, a2 offset:65535

// GFX90A: ds_sub_rtn_u32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x42,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_sub_rtn_u32 a5, v255, a2 offset:65535

// GFX90A: ds_sub_rtn_u32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x42,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_sub_rtn_u32 a5, v1, a255 offset:65535

// GFX90A: ds_sub_rtn_u32 a5, v1, a2       ; encoding: [0x00,0x00,0x42,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_sub_rtn_u32 a5, v1, a2

// GFX90A: ds_sub_rtn_u32 a5, v1, a2       ; encoding: [0x00,0x00,0x42,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_sub_rtn_u32 a5, v1, a2

// GFX90A: ds_sub_rtn_u32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x42,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_sub_rtn_u32 a5, v1, a2 offset:4

// GFX90A: ds_rsub_rtn_u32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x44,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_rsub_rtn_u32 a5, v1, a2 offset:65535

// GFX90A: ds_rsub_rtn_u32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x44,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_rsub_rtn_u32 a255, v1, a2 offset:65535

// GFX90A: ds_rsub_rtn_u32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x44,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_rsub_rtn_u32 a5, v255, a2 offset:65535

// GFX90A: ds_rsub_rtn_u32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x44,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_rsub_rtn_u32 a5, v1, a255 offset:65535

// GFX90A: ds_rsub_rtn_u32 a5, v1, a2      ; encoding: [0x00,0x00,0x44,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_rsub_rtn_u32 a5, v1, a2

// GFX90A: ds_rsub_rtn_u32 a5, v1, a2      ; encoding: [0x00,0x00,0x44,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_rsub_rtn_u32 a5, v1, a2

// GFX90A: ds_rsub_rtn_u32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x44,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_rsub_rtn_u32 a5, v1, a2 offset:4

// GFX90A: ds_inc_rtn_u32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x46,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_inc_rtn_u32 a5, v1, a2 offset:65535

// GFX90A: ds_inc_rtn_u32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x46,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_inc_rtn_u32 a255, v1, a2 offset:65535

// GFX90A: ds_inc_rtn_u32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x46,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_inc_rtn_u32 a5, v255, a2 offset:65535

// GFX90A: ds_inc_rtn_u32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x46,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_inc_rtn_u32 a5, v1, a255 offset:65535

// GFX90A: ds_inc_rtn_u32 a5, v1, a2       ; encoding: [0x00,0x00,0x46,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_inc_rtn_u32 a5, v1, a2

// GFX90A: ds_inc_rtn_u32 a5, v1, a2       ; encoding: [0x00,0x00,0x46,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_inc_rtn_u32 a5, v1, a2

// GFX90A: ds_inc_rtn_u32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x46,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_inc_rtn_u32 a5, v1, a2 offset:4

// GFX90A: ds_dec_rtn_u32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x48,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_dec_rtn_u32 a5, v1, a2 offset:65535

// GFX90A: ds_dec_rtn_u32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x48,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_dec_rtn_u32 a255, v1, a2 offset:65535

// GFX90A: ds_dec_rtn_u32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x48,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_dec_rtn_u32 a5, v255, a2 offset:65535

// GFX90A: ds_dec_rtn_u32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x48,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_dec_rtn_u32 a5, v1, a255 offset:65535

// GFX90A: ds_dec_rtn_u32 a5, v1, a2       ; encoding: [0x00,0x00,0x48,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_dec_rtn_u32 a5, v1, a2

// GFX90A: ds_dec_rtn_u32 a5, v1, a2       ; encoding: [0x00,0x00,0x48,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_dec_rtn_u32 a5, v1, a2

// GFX90A: ds_dec_rtn_u32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x48,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_dec_rtn_u32 a5, v1, a2 offset:4

// GFX90A: ds_min_rtn_i32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x4a,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_i32 a5, v1, a2 offset:65535

// GFX90A: ds_min_rtn_i32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x4a,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_i32 a255, v1, a2 offset:65535

// GFX90A: ds_min_rtn_i32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x4a,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_i32 a5, v255, a2 offset:65535

// GFX90A: ds_min_rtn_i32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x4a,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_i32 a5, v1, a255 offset:65535

// GFX90A: ds_min_rtn_i32 a5, v1, a2       ; encoding: [0x00,0x00,0x4a,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_i32 a5, v1, a2

// GFX90A: ds_min_rtn_i32 a5, v1, a2       ; encoding: [0x00,0x00,0x4a,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_i32 a5, v1, a2

// GFX90A: ds_min_rtn_i32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x4a,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_i32 a5, v1, a2 offset:4

// GFX90A: ds_max_rtn_i32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x4c,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_i32 a5, v1, a2 offset:65535

// GFX90A: ds_max_rtn_i32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x4c,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_i32 a255, v1, a2 offset:65535

// GFX90A: ds_max_rtn_i32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x4c,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_i32 a5, v255, a2 offset:65535

// GFX90A: ds_max_rtn_i32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x4c,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_i32 a5, v1, a255 offset:65535

// GFX90A: ds_max_rtn_i32 a5, v1, a2       ; encoding: [0x00,0x00,0x4c,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_i32 a5, v1, a2

// GFX90A: ds_max_rtn_i32 a5, v1, a2       ; encoding: [0x00,0x00,0x4c,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_i32 a5, v1, a2

// GFX90A: ds_max_rtn_i32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x4c,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_i32 a5, v1, a2 offset:4

// GFX90A: ds_min_rtn_u32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x4e,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_u32 a5, v1, a2 offset:65535

// GFX90A: ds_min_rtn_u32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x4e,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_u32 a255, v1, a2 offset:65535

// GFX90A: ds_min_rtn_u32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x4e,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_u32 a5, v255, a2 offset:65535

// GFX90A: ds_min_rtn_u32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x4e,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_u32 a5, v1, a255 offset:65535

// GFX90A: ds_min_rtn_u32 a5, v1, a2       ; encoding: [0x00,0x00,0x4e,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_u32 a5, v1, a2

// GFX90A: ds_min_rtn_u32 a5, v1, a2       ; encoding: [0x00,0x00,0x4e,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_u32 a5, v1, a2

// GFX90A: ds_min_rtn_u32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x4e,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_u32 a5, v1, a2 offset:4

// GFX90A: ds_max_rtn_u32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x50,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_u32 a5, v1, a2 offset:65535

// GFX90A: ds_max_rtn_u32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x50,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_u32 a255, v1, a2 offset:65535

// GFX90A: ds_max_rtn_u32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x50,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_u32 a5, v255, a2 offset:65535

// GFX90A: ds_max_rtn_u32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x50,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_u32 a5, v1, a255 offset:65535

// GFX90A: ds_max_rtn_u32 a5, v1, a2       ; encoding: [0x00,0x00,0x50,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_u32 a5, v1, a2

// GFX90A: ds_max_rtn_u32 a5, v1, a2       ; encoding: [0x00,0x00,0x50,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_u32 a5, v1, a2

// GFX90A: ds_max_rtn_u32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x50,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_u32 a5, v1, a2 offset:4

// GFX90A: ds_and_rtn_b32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x52,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_and_rtn_b32 a5, v1, a2 offset:65535

// GFX90A: ds_and_rtn_b32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x52,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_and_rtn_b32 a255, v1, a2 offset:65535

// GFX90A: ds_and_rtn_b32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x52,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_and_rtn_b32 a5, v255, a2 offset:65535

// GFX90A: ds_and_rtn_b32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x52,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_and_rtn_b32 a5, v1, a255 offset:65535

// GFX90A: ds_and_rtn_b32 a5, v1, a2       ; encoding: [0x00,0x00,0x52,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_and_rtn_b32 a5, v1, a2

// GFX90A: ds_and_rtn_b32 a5, v1, a2       ; encoding: [0x00,0x00,0x52,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_and_rtn_b32 a5, v1, a2

// GFX90A: ds_and_rtn_b32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x52,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_and_rtn_b32 a5, v1, a2 offset:4

// GFX90A: ds_or_rtn_b32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x54,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_or_rtn_b32 a5, v1, a2 offset:65535

// GFX90A: ds_or_rtn_b32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x54,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_or_rtn_b32 a255, v1, a2 offset:65535

// GFX90A: ds_or_rtn_b32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x54,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_or_rtn_b32 a5, v255, a2 offset:65535

// GFX90A: ds_or_rtn_b32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x54,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_or_rtn_b32 a5, v1, a255 offset:65535

// GFX90A: ds_or_rtn_b32 a5, v1, a2        ; encoding: [0x00,0x00,0x54,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_or_rtn_b32 a5, v1, a2

// GFX90A: ds_or_rtn_b32 a5, v1, a2        ; encoding: [0x00,0x00,0x54,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_or_rtn_b32 a5, v1, a2

// GFX90A: ds_or_rtn_b32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x54,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_or_rtn_b32 a5, v1, a2 offset:4

// GFX90A: ds_xor_rtn_b32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x56,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_xor_rtn_b32 a5, v1, a2 offset:65535

// GFX90A: ds_xor_rtn_b32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x56,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_xor_rtn_b32 a255, v1, a2 offset:65535

// GFX90A: ds_xor_rtn_b32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x56,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_xor_rtn_b32 a5, v255, a2 offset:65535

// GFX90A: ds_xor_rtn_b32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x56,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_xor_rtn_b32 a5, v1, a255 offset:65535

// GFX90A: ds_xor_rtn_b32 a5, v1, a2       ; encoding: [0x00,0x00,0x56,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_xor_rtn_b32 a5, v1, a2

// GFX90A: ds_xor_rtn_b32 a5, v1, a2       ; encoding: [0x00,0x00,0x56,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_xor_rtn_b32 a5, v1, a2

// GFX90A: ds_xor_rtn_b32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x56,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_xor_rtn_b32 a5, v1, a2 offset:4

// GFX90A: ds_mskor_rtn_b32 a5, v1, a2, a5 offset:65535 ; encoding: [0xff,0xff,0x58,0xda,0x01,0x02,0x05,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_rtn_b32 a5, v1, a2, a5 offset:65535

// GFX90A: ds_mskor_rtn_b32 a255, v1, a2, a5 offset:65535 ; encoding: [0xff,0xff,0x58,0xda,0x01,0x02,0x05,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_rtn_b32 a255, v1, a2, a5 offset:65535

// GFX90A: ds_mskor_rtn_b32 a5, v255, a2, a5 offset:65535 ; encoding: [0xff,0xff,0x58,0xda,0xff,0x02,0x05,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_rtn_b32 a5, v255, a2, a5 offset:65535

// GFX90A: ds_mskor_rtn_b32 a5, v1, a255, a3 offset:65535 ; encoding: [0xff,0xff,0x58,0xda,0x01,0xff,0x03,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_rtn_b32 a5, v1, a255, a3 offset:65535

// GFX90A: ds_mskor_rtn_b32 a5, v1, a2, a5 offset:65535 ; encoding: [0xff,0xff,0x58,0xda,0x01,0x02,0x05,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_rtn_b32 a5, v1, a2, a5 offset:65535

// GFX90A: ds_mskor_rtn_b32 a5, v1, a2, a5 ; encoding: [0x00,0x00,0x58,0xda,0x01,0x02,0x05,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_rtn_b32 a5, v1, a2, a5

// GFX90A: ds_mskor_rtn_b32 a5, v1, a2, a5 ; encoding: [0x00,0x00,0x58,0xda,0x01,0x02,0x05,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_rtn_b32 a5, v1, a2, a5

// GFX90A: ds_mskor_rtn_b32 a5, v1, a2, a5 offset:4 ; encoding: [0x04,0x00,0x58,0xda,0x01,0x02,0x05,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_rtn_b32 a5, v1, a2, a5 offset:4

// GFX90A: ds_wrxchg_rtn_b32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x5a,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg_rtn_b32 a5, v1, a2 offset:65535

// GFX90A: ds_wrxchg_rtn_b32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x5a,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg_rtn_b32 a255, v1, a2 offset:65535

// GFX90A: ds_wrxchg_rtn_b32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x5a,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg_rtn_b32 a5, v255, a2 offset:65535

// GFX90A: ds_wrxchg_rtn_b32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x5a,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg_rtn_b32 a5, v1, a255 offset:65535

// GFX90A: ds_wrxchg_rtn_b32 a5, v1, a2    ; encoding: [0x00,0x00,0x5a,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg_rtn_b32 a5, v1, a2

// GFX90A: ds_wrxchg_rtn_b32 a5, v1, a2    ; encoding: [0x00,0x00,0x5a,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg_rtn_b32 a5, v1, a2

// GFX90A: ds_wrxchg_rtn_b32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x5a,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg_rtn_b32 a5, v1, a2 offset:4

// GFX90A: ds_wrxchg2_rtn_b32 a[6:7], v1, a2, a3 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x5c,0xda,0x01,0x02,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b32 a[6:7], v1, a2, a3 offset0:127 offset1:255

// GFX90A: ds_wrxchg2_rtn_b32 a[254:255], v1, a2, a3 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x5c,0xda,0x01,0x02,0x03,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b32 a[254:255], v1, a2, a3 offset0:127 offset1:255

// GFX90A: ds_wrxchg2_rtn_b32 a[6:7], v255, a2, a3 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x5c,0xda,0xff,0x02,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b32 a[6:7], v255, a2, a3 offset0:127 offset1:255

// GFX90A: ds_wrxchg2_rtn_b32 a[6:7], v1, a255, a3 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x5c,0xda,0x01,0xff,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b32 a[6:7], v1, a255, a3 offset0:127 offset1:255

// GFX90A: ds_wrxchg2_rtn_b32 a[6:7], v1, a2, a255 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x5c,0xda,0x01,0x02,0xff,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b32 a[6:7], v1, a2, a255 offset0:127 offset1:255

// GFX90A: ds_wrxchg2_rtn_b32 a[6:7], v1, a2, a3 offset1:255 ; encoding: [0x00,0xff,0x5c,0xda,0x01,0x02,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b32 a[6:7], v1, a2, a3 offset1:255

// GFX90A: ds_wrxchg2_rtn_b32 a[6:7], v1, a2, a3 offset1:255 ; encoding: [0x00,0xff,0x5c,0xda,0x01,0x02,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b32 a[6:7], v1, a2, a3 offset1:255

// GFX90A: ds_wrxchg2_rtn_b32 a[6:7], v1, a2, a3 offset0:16 offset1:255 ; encoding: [0x10,0xff,0x5c,0xda,0x01,0x02,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b32 a[6:7], v1, a2, a3 offset0:16 offset1:255

// GFX90A: ds_wrxchg2_rtn_b32 a[6:7], v1, a2, a3 offset0:127 ; encoding: [0x7f,0x00,0x5c,0xda,0x01,0x02,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b32 a[6:7], v1, a2, a3 offset0:127

// GFX90A: ds_wrxchg2_rtn_b32 a[6:7], v1, a2, a3 offset0:127 ; encoding: [0x7f,0x00,0x5c,0xda,0x01,0x02,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b32 a[6:7], v1, a2, a3 offset0:127

// GFX90A: ds_wrxchg2_rtn_b32 a[6:7], v1, a2, a3 offset0:127 offset1:1 ; encoding: [0x7f,0x01,0x5c,0xda,0x01,0x02,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b32 a[6:7], v1, a2, a3 offset0:127 offset1:1

// GFX90A: ds_wrxchg2st64_rtn_b32 a[6:7], v1, a2, a3 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x5e,0xda,0x01,0x02,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b32 a[6:7], v1, a2, a3 offset0:127 offset1:255

// GFX90A: ds_wrxchg2st64_rtn_b32 a[254:255], v1, a2, a3 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x5e,0xda,0x01,0x02,0x03,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b32 a[254:255], v1, a2, a3 offset0:127 offset1:255

// GFX90A: ds_wrxchg2st64_rtn_b32 a[6:7], v255, a2, a3 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x5e,0xda,0xff,0x02,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b32 a[6:7], v255, a2, a3 offset0:127 offset1:255

// GFX90A: ds_wrxchg2st64_rtn_b32 a[6:7], v1, a255, a3 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x5e,0xda,0x01,0xff,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b32 a[6:7], v1, a255, a3 offset0:127 offset1:255

// GFX90A: ds_wrxchg2st64_rtn_b32 a[6:7], v1, a2, a255 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x5e,0xda,0x01,0x02,0xff,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b32 a[6:7], v1, a2, a255 offset0:127 offset1:255

// GFX90A: ds_wrxchg2st64_rtn_b32 a[6:7], v1, a2, a3 offset1:255 ; encoding: [0x00,0xff,0x5e,0xda,0x01,0x02,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b32 a[6:7], v1, a2, a3 offset1:255

// GFX90A: ds_wrxchg2st64_rtn_b32 a[6:7], v1, a2, a3 offset1:255 ; encoding: [0x00,0xff,0x5e,0xda,0x01,0x02,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b32 a[6:7], v1, a2, a3 offset1:255

// GFX90A: ds_wrxchg2st64_rtn_b32 a[6:7], v1, a2, a3 offset0:16 offset1:255 ; encoding: [0x10,0xff,0x5e,0xda,0x01,0x02,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b32 a[6:7], v1, a2, a3 offset0:16 offset1:255

// GFX90A: ds_wrxchg2st64_rtn_b32 a[6:7], v1, a2, a3 offset0:127 ; encoding: [0x7f,0x00,0x5e,0xda,0x01,0x02,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b32 a[6:7], v1, a2, a3 offset0:127

// GFX90A: ds_wrxchg2st64_rtn_b32 a[6:7], v1, a2, a3 offset0:127 ; encoding: [0x7f,0x00,0x5e,0xda,0x01,0x02,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b32 a[6:7], v1, a2, a3 offset0:127

// GFX90A: ds_wrxchg2st64_rtn_b32 a[6:7], v1, a2, a3 offset0:127 offset1:1 ; encoding: [0x7f,0x01,0x5e,0xda,0x01,0x02,0x03,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b32 a[6:7], v1, a2, a3 offset0:127 offset1:1

// GFX90A: ds_cmpst_rtn_b32 a5, v1, a2, a3 offset:65535 ; encoding: [0xff,0xff,0x60,0xda,0x01,0x02,0x03,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_b32 a5, v1, a2, a3 offset:65535

// GFX90A: ds_cmpst_rtn_b32 a255, v1, a2, a3 offset:65535 ; encoding: [0xff,0xff,0x60,0xda,0x01,0x02,0x03,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_b32 a255, v1, a2, a3 offset:65535

// GFX90A: ds_cmpst_rtn_b32 a5, v255, a2, a3 offset:65535 ; encoding: [0xff,0xff,0x60,0xda,0xff,0x02,0x03,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_b32 a5, v255, a2, a3 offset:65535

// GFX90A: ds_cmpst_rtn_b32 a5, v1, a255, a3 offset:65535 ; encoding: [0xff,0xff,0x60,0xda,0x01,0xff,0x03,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_b32 a5, v1, a255, a3 offset:65535

// GFX90A: ds_cmpst_rtn_b32 a5, v1, a2, a255 offset:65535 ; encoding: [0xff,0xff,0x60,0xda,0x01,0x02,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_b32 a5, v1, a2, a255 offset:65535

// GFX90A: ds_cmpst_rtn_b32 a5, v1, a2, a3 ; encoding: [0x00,0x00,0x60,0xda,0x01,0x02,0x03,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_b32 a5, v1, a2, a3

// GFX90A: ds_cmpst_rtn_b32 a5, v1, a2, a3 ; encoding: [0x00,0x00,0x60,0xda,0x01,0x02,0x03,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_b32 a5, v1, a2, a3

// GFX90A: ds_cmpst_rtn_b32 a5, v1, a2, a3 offset:4 ; encoding: [0x04,0x00,0x60,0xda,0x01,0x02,0x03,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_b32 a5, v1, a2, a3 offset:4

// GFX90A: ds_cmpst_rtn_f32 a5, v1, a2, a3 offset:65535 ; encoding: [0xff,0xff,0x62,0xda,0x01,0x02,0x03,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_f32 a5, v1, a2, a3 offset:65535

// GFX90A: ds_cmpst_rtn_f32 a255, v1, a2, a3 offset:65535 ; encoding: [0xff,0xff,0x62,0xda,0x01,0x02,0x03,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_f32 a255, v1, a2, a3 offset:65535

// GFX90A: ds_cmpst_rtn_f32 a5, v255, a2, a3 offset:65535 ; encoding: [0xff,0xff,0x62,0xda,0xff,0x02,0x03,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_f32 a5, v255, a2, a3 offset:65535

// GFX90A: ds_cmpst_rtn_f32 a5, v1, a255, a3 offset:65535 ; encoding: [0xff,0xff,0x62,0xda,0x01,0xff,0x03,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_f32 a5, v1, a255, a3 offset:65535

// GFX90A: ds_cmpst_rtn_f32 a5, v1, a2, a255 offset:65535 ; encoding: [0xff,0xff,0x62,0xda,0x01,0x02,0xff,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_f32 a5, v1, a2, a255 offset:65535

// GFX90A: ds_cmpst_rtn_f32 a5, v1, a2, a3 ; encoding: [0x00,0x00,0x62,0xda,0x01,0x02,0x03,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_f32 a5, v1, a2, a3

// GFX90A: ds_cmpst_rtn_f32 a5, v1, a2, a3 ; encoding: [0x00,0x00,0x62,0xda,0x01,0x02,0x03,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_f32 a5, v1, a2, a3

// GFX90A: ds_cmpst_rtn_f32 a5, v1, a2, a3 offset:4 ; encoding: [0x04,0x00,0x62,0xda,0x01,0x02,0x03,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_f32 a5, v1, a2, a3 offset:4

// GFX90A: ds_min_rtn_f32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x64,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_f32 a5, v1, a2 offset:65535

// GFX90A: ds_min_rtn_f32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x64,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_f32 a255, v1, a2 offset:65535

// GFX90A: ds_min_rtn_f32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x64,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_f32 a5, v255, a2 offset:65535

// GFX90A: ds_min_rtn_f32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x64,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_f32 a5, v1, a255 offset:65535

// GFX90A: ds_min_rtn_f32 a5, v1, a2       ; encoding: [0x00,0x00,0x64,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_f32 a5, v1, a2

// GFX90A: ds_min_rtn_f32 a5, v1, a2       ; encoding: [0x00,0x00,0x64,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_f32 a5, v1, a2

// GFX90A: ds_min_rtn_f32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x64,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_f32 a5, v1, a2 offset:4

// GFX90A: ds_max_rtn_f32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x66,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_f32 a5, v1, a2 offset:65535

// GFX90A: ds_max_rtn_f32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x66,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_f32 a255, v1, a2 offset:65535

// GFX90A: ds_max_rtn_f32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x66,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_f32 a5, v255, a2 offset:65535

// GFX90A: ds_max_rtn_f32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x66,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_f32 a5, v1, a255 offset:65535

// GFX90A: ds_max_rtn_f32 a5, v1, a2       ; encoding: [0x00,0x00,0x66,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_f32 a5, v1, a2

// GFX90A: ds_max_rtn_f32 a5, v1, a2       ; encoding: [0x00,0x00,0x66,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_f32 a5, v1, a2

// GFX90A: ds_max_rtn_f32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x66,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_f32 a5, v1, a2 offset:4

// GFX90A: ds_wrap_rtn_b32 a5, v1, a2, a5 offset:65535 ; encoding: [0xff,0xff,0x68,0xda,0x01,0x02,0x05,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrap_rtn_b32 a5, v1, a2, a5 offset:65535

// GFX90A: ds_wrap_rtn_b32 a255, v1, a2, a5 offset:65535 ; encoding: [0xff,0xff,0x68,0xda,0x01,0x02,0x05,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrap_rtn_b32 a255, v1, a2, a5 offset:65535

// GFX90A: ds_wrap_rtn_b32 a5, v255, a2, a5 offset:65535 ; encoding: [0xff,0xff,0x68,0xda,0xff,0x02,0x05,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrap_rtn_b32 a5, v255, a2, a5 offset:65535

// GFX90A: ds_wrap_rtn_b32 a5, v1, a255, a3 offset:65535 ; encoding: [0xff,0xff,0x68,0xda,0x01,0xff,0x03,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrap_rtn_b32 a5, v1, a255, a3 offset:65535

// GFX90A: ds_wrap_rtn_b32 a5, v1, a2, a5 offset:65535 ; encoding: [0xff,0xff,0x68,0xda,0x01,0x02,0x05,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrap_rtn_b32 a5, v1, a2, a5 offset:65535

// GFX90A: ds_wrap_rtn_b32 a5, v1, a2, a5  ; encoding: [0x00,0x00,0x68,0xda,0x01,0x02,0x05,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrap_rtn_b32 a5, v1, a2, a5

// GFX90A: ds_wrap_rtn_b32 a5, v1, a2, a5  ; encoding: [0x00,0x00,0x68,0xda,0x01,0x02,0x05,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrap_rtn_b32 a5, v1, a2, a5

// GFX90A: ds_wrap_rtn_b32 a5, v1, a2, a5 offset:4 ; encoding: [0x04,0x00,0x68,0xda,0x01,0x02,0x05,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrap_rtn_b32 a5, v1, a2, a5 offset:4

// GFX90A: ds_add_rtn_f32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x6a,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_f32 a5, v1, a2 offset:65535

// GFX90A: ds_add_rtn_f32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x6a,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_f32 a255, v1, a2 offset:65535

// GFX90A: ds_add_rtn_f32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x6a,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_f32 a5, v255, a2 offset:65535

// GFX90A: ds_add_rtn_f32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x6a,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_f32 a5, v1, a255 offset:65535

// GFX90A: ds_add_rtn_f32 a5, v1, a2       ; encoding: [0x00,0x00,0x6a,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_f32 a5, v1, a2

// GFX90A: ds_add_rtn_f32 a5, v1, a2       ; encoding: [0x00,0x00,0x6a,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_f32 a5, v1, a2

// GFX90A: ds_add_rtn_f32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x6a,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_f32 a5, v1, a2 offset:4

// GFX90A: ds_read_b32 a5, v1 offset:65535 ; encoding: [0xff,0xff,0x6c,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b32 a5, v1 offset:65535

// GFX90A: ds_read_b32 a255, v1 offset:65535 ; encoding: [0xff,0xff,0x6c,0xda,0x01,0x00,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b32 a255, v1 offset:65535

// GFX90A: ds_read_b32 a5, v255 offset:65535 ; encoding: [0xff,0xff,0x6c,0xda,0xff,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b32 a5, v255 offset:65535

// GFX90A: ds_read_b32 a5, v1              ; encoding: [0x00,0x00,0x6c,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b32 a5, v1

// GFX90A: ds_read_b32 a5, v1              ; encoding: [0x00,0x00,0x6c,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b32 a5, v1

// GFX90A: ds_read_b32 a5, v1 offset:4     ; encoding: [0x04,0x00,0x6c,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b32 a5, v1 offset:4

// GFX90A: ds_read2_b32 a[6:7], v1 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x6e,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b32 a[6:7], v1 offset0:127 offset1:255

// GFX90A: ds_read2_b32 a[254:255], v1 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x6e,0xda,0x01,0x00,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b32 a[254:255], v1 offset0:127 offset1:255

// GFX90A: ds_read2_b32 a[6:7], v255 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x6e,0xda,0xff,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b32 a[6:7], v255 offset0:127 offset1:255

// GFX90A: ds_read2_b32 a[6:7], v1 offset1:255 ; encoding: [0x00,0xff,0x6e,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b32 a[6:7], v1 offset1:255

// GFX90A: ds_read2_b32 a[6:7], v1 offset1:255 ; encoding: [0x00,0xff,0x6e,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b32 a[6:7], v1 offset1:255

// GFX90A: ds_read2_b32 a[6:7], v1 offset0:16 offset1:255 ; encoding: [0x10,0xff,0x6e,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b32 a[6:7], v1 offset0:16 offset1:255

// GFX90A: ds_read2_b32 a[6:7], v1 offset0:127 ; encoding: [0x7f,0x00,0x6e,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b32 a[6:7], v1 offset0:127

// GFX90A: ds_read2_b32 a[6:7], v1 offset0:127 ; encoding: [0x7f,0x00,0x6e,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b32 a[6:7], v1 offset0:127

// GFX90A: ds_read2_b32 a[6:7], v1 offset0:127 offset1:1 ; encoding: [0x7f,0x01,0x6e,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b32 a[6:7], v1 offset0:127 offset1:1

// GFX90A: ds_read2st64_b32 a[6:7], v1 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x70,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b32 a[6:7], v1 offset0:127 offset1:255

// GFX90A: ds_read2st64_b32 a[254:255], v1 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x70,0xda,0x01,0x00,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b32 a[254:255], v1 offset0:127 offset1:255

// GFX90A: ds_read2st64_b32 a[6:7], v255 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x70,0xda,0xff,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b32 a[6:7], v255 offset0:127 offset1:255

// GFX90A: ds_read2st64_b32 a[6:7], v1 offset1:255 ; encoding: [0x00,0xff,0x70,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b32 a[6:7], v1 offset1:255

// GFX90A: ds_read2st64_b32 a[6:7], v1 offset1:255 ; encoding: [0x00,0xff,0x70,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b32 a[6:7], v1 offset1:255

// GFX90A: ds_read2st64_b32 a[6:7], v1 offset0:16 offset1:255 ; encoding: [0x10,0xff,0x70,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b32 a[6:7], v1 offset0:16 offset1:255

// GFX90A: ds_read2st64_b32 a[6:7], v1 offset0:127 ; encoding: [0x7f,0x00,0x70,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b32 a[6:7], v1 offset0:127

// GFX90A: ds_read2st64_b32 a[6:7], v1 offset0:127 ; encoding: [0x7f,0x00,0x70,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b32 a[6:7], v1 offset0:127

// GFX90A: ds_read2st64_b32 a[6:7], v1 offset0:127 offset1:1 ; encoding: [0x7f,0x01,0x70,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b32 a[6:7], v1 offset0:127 offset1:1

// GFX90A: ds_read_i8 a5, v1 offset:65535  ; encoding: [0xff,0xff,0x72,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8 a5, v1 offset:65535

// GFX90A: ds_read_i8 a255, v1 offset:65535 ; encoding: [0xff,0xff,0x72,0xda,0x01,0x00,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8 a255, v1 offset:65535

// GFX90A: ds_read_i8 a5, v255 offset:65535 ; encoding: [0xff,0xff,0x72,0xda,0xff,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8 a5, v255 offset:65535

// GFX90A: ds_read_i8 a5, v1               ; encoding: [0x00,0x00,0x72,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8 a5, v1

// GFX90A: ds_read_i8 a5, v1               ; encoding: [0x00,0x00,0x72,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8 a5, v1

// GFX90A: ds_read_i8 a5, v1 offset:4      ; encoding: [0x04,0x00,0x72,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8 a5, v1 offset:4

// GFX90A: ds_read_u8 a5, v1 offset:65535  ; encoding: [0xff,0xff,0x74,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8 a5, v1 offset:65535

// GFX90A: ds_read_u8 a255, v1 offset:65535 ; encoding: [0xff,0xff,0x74,0xda,0x01,0x00,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8 a255, v1 offset:65535

// GFX90A: ds_read_u8 a5, v255 offset:65535 ; encoding: [0xff,0xff,0x74,0xda,0xff,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8 a5, v255 offset:65535

// GFX90A: ds_read_u8 a5, v1               ; encoding: [0x00,0x00,0x74,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8 a5, v1

// GFX90A: ds_read_u8 a5, v1               ; encoding: [0x00,0x00,0x74,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8 a5, v1

// GFX90A: ds_read_u8 a5, v1 offset:4      ; encoding: [0x04,0x00,0x74,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8 a5, v1 offset:4

// GFX90A: ds_read_i16 a5, v1 offset:65535 ; encoding: [0xff,0xff,0x76,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i16 a5, v1 offset:65535

// GFX90A: ds_read_i16 a255, v1 offset:65535 ; encoding: [0xff,0xff,0x76,0xda,0x01,0x00,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i16 a255, v1 offset:65535

// GFX90A: ds_read_i16 a5, v255 offset:65535 ; encoding: [0xff,0xff,0x76,0xda,0xff,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i16 a5, v255 offset:65535

// GFX90A: ds_read_i16 a5, v1              ; encoding: [0x00,0x00,0x76,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i16 a5, v1

// GFX90A: ds_read_i16 a5, v1              ; encoding: [0x00,0x00,0x76,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i16 a5, v1

// GFX90A: ds_read_i16 a5, v1 offset:4     ; encoding: [0x04,0x00,0x76,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i16 a5, v1 offset:4

// GFX90A: ds_read_u16 a5, v1 offset:65535 ; encoding: [0xff,0xff,0x78,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16 a5, v1 offset:65535

// GFX90A: ds_read_u16 a255, v1 offset:65535 ; encoding: [0xff,0xff,0x78,0xda,0x01,0x00,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16 a255, v1 offset:65535

// GFX90A: ds_read_u16 a5, v255 offset:65535 ; encoding: [0xff,0xff,0x78,0xda,0xff,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16 a5, v255 offset:65535

// GFX90A: ds_read_u16 a5, v1              ; encoding: [0x00,0x00,0x78,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16 a5, v1

// GFX90A: ds_read_u16 a5, v1              ; encoding: [0x00,0x00,0x78,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16 a5, v1

// GFX90A: ds_read_u16 a5, v1 offset:4     ; encoding: [0x04,0x00,0x78,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16 a5, v1 offset:4

// GFX90A: ds_swizzle_b32 a5, v1 offset:swizzle(FFT,31) ; encoding: [0xff,0xff,0x7a,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_swizzle_b32 a5, v1 offset:65535

// GFX90A: ds_swizzle_b32 a255, v1 offset:swizzle(FFT,31) ; encoding: [0xff,0xff,0x7a,0xda,0x01,0x00,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_swizzle_b32 a255, v1 offset:65535

// GFX90A: ds_swizzle_b32 a5, v255 offset:swizzle(FFT,31) ; encoding: [0xff,0xff,0x7a,0xda,0xff,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_swizzle_b32 a5, v255 offset:65535

// GFX90A: ds_swizzle_b32 a5, v1           ; encoding: [0x00,0x00,0x7a,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_swizzle_b32 a5, v1

// GFX90A: ds_swizzle_b32 a5, v1           ; encoding: [0x00,0x00,0x7a,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_swizzle_b32 a5, v1

// GFX90A: ds_swizzle_b32 a5, v1 offset:swizzle(BITMASK_PERM,"00p00") ; encoding: [0x04,0x00,0x7a,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_swizzle_b32 a5, v1 offset:swizzle(BITMASK_PERM,"00p00")

// GFX90A: ds_permute_b32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x7c,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_permute_b32 a5, v1, a2 offset:65535

// GFX90A: ds_permute_b32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x7c,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_permute_b32 a255, v1, a2 offset:65535

// GFX90A: ds_permute_b32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x7c,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_permute_b32 a5, v255, a2 offset:65535

// GFX90A: ds_permute_b32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x7c,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_permute_b32 a5, v1, a255 offset:65535

// GFX90A: ds_permute_b32 a5, v1, a2       ; encoding: [0x00,0x00,0x7c,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_permute_b32 a5, v1, a2

// GFX90A: ds_permute_b32 a5, v1, a2       ; encoding: [0x00,0x00,0x7c,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_permute_b32 a5, v1, a2

// GFX90A: ds_permute_b32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x7c,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_permute_b32 a5, v1, a2 offset:4

// GFX90A: ds_bpermute_b32 a5, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x7e,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_bpermute_b32 a5, v1, a2 offset:65535

// GFX90A: ds_bpermute_b32 a255, v1, a2 offset:65535 ; encoding: [0xff,0xff,0x7e,0xda,0x01,0x02,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_bpermute_b32 a255, v1, a2 offset:65535

// GFX90A: ds_bpermute_b32 a5, v255, a2 offset:65535 ; encoding: [0xff,0xff,0x7e,0xda,0xff,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_bpermute_b32 a5, v255, a2 offset:65535

// GFX90A: ds_bpermute_b32 a5, v1, a255 offset:65535 ; encoding: [0xff,0xff,0x7e,0xda,0x01,0xff,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_bpermute_b32 a5, v1, a255 offset:65535

// GFX90A: ds_bpermute_b32 a5, v1, a2      ; encoding: [0x00,0x00,0x7e,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_bpermute_b32 a5, v1, a2

// GFX90A: ds_bpermute_b32 a5, v1, a2      ; encoding: [0x00,0x00,0x7e,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_bpermute_b32 a5, v1, a2

// GFX90A: ds_bpermute_b32 a5, v1, a2 offset:4 ; encoding: [0x04,0x00,0x7e,0xda,0x01,0x02,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_bpermute_b32 a5, v1, a2 offset:4

// GFX90A: ds_add_u64 v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x80,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_u64 v1, a[2:3] offset:65535

// GFX90A: ds_add_u64 v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x80,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_u64 v255, a[2:3] offset:65535

// GFX90A: ds_add_u64 v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0x80,0xda,0x01,0xfe,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_u64 v1, a[254:255] offset:65535

// GFX90A: ds_add_u64 v1, a[2:3]           ; encoding: [0x00,0x00,0x80,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_u64 v1, a[2:3]

// GFX90A: ds_add_u64 v1, a[2:3]           ; encoding: [0x00,0x00,0x80,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_u64 v1, a[2:3]

// GFX90A: ds_add_u64 v1, a[2:3] offset:4  ; encoding: [0x04,0x00,0x80,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_add_u64 v1, a[2:3] offset:4

// GFX90A: ds_sub_u64 v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x82,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_sub_u64 v1, a[2:3] offset:65535

// GFX90A: ds_sub_u64 v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x82,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_sub_u64 v255, a[2:3] offset:65535

// GFX90A: ds_sub_u64 v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0x82,0xda,0x01,0xfe,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_sub_u64 v1, a[254:255] offset:65535

// GFX90A: ds_sub_u64 v1, a[2:3]           ; encoding: [0x00,0x00,0x82,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_sub_u64 v1, a[2:3]

// GFX90A: ds_sub_u64 v1, a[2:3]           ; encoding: [0x00,0x00,0x82,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_sub_u64 v1, a[2:3]

// GFX90A: ds_sub_u64 v1, a[2:3] offset:4  ; encoding: [0x04,0x00,0x82,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_sub_u64 v1, a[2:3] offset:4

// GFX90A: ds_rsub_u64 v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x84,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_rsub_u64 v1, a[2:3] offset:65535

// GFX90A: ds_rsub_u64 v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x84,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_rsub_u64 v255, a[2:3] offset:65535

// GFX90A: ds_rsub_u64 v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0x84,0xda,0x01,0xfe,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_rsub_u64 v1, a[254:255] offset:65535

// GFX90A: ds_rsub_u64 v1, a[2:3]          ; encoding: [0x00,0x00,0x84,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_rsub_u64 v1, a[2:3]

// GFX90A: ds_rsub_u64 v1, a[2:3]          ; encoding: [0x00,0x00,0x84,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_rsub_u64 v1, a[2:3]

// GFX90A: ds_rsub_u64 v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0x84,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_rsub_u64 v1, a[2:3] offset:4

// GFX90A: ds_inc_u64 v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x86,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_inc_u64 v1, a[2:3] offset:65535

// GFX90A: ds_inc_u64 v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x86,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_inc_u64 v255, a[2:3] offset:65535

// GFX90A: ds_inc_u64 v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0x86,0xda,0x01,0xfe,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_inc_u64 v1, a[254:255] offset:65535

// GFX90A: ds_inc_u64 v1, a[2:3]           ; encoding: [0x00,0x00,0x86,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_inc_u64 v1, a[2:3]

// GFX90A: ds_inc_u64 v1, a[2:3]           ; encoding: [0x00,0x00,0x86,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_inc_u64 v1, a[2:3]

// GFX90A: ds_inc_u64 v1, a[2:3] offset:4  ; encoding: [0x04,0x00,0x86,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_inc_u64 v1, a[2:3] offset:4

// GFX90A: ds_dec_u64 v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x88,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_dec_u64 v1, a[2:3] offset:65535

// GFX90A: ds_dec_u64 v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x88,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_dec_u64 v255, a[2:3] offset:65535

// GFX90A: ds_dec_u64 v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0x88,0xda,0x01,0xfe,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_dec_u64 v1, a[254:255] offset:65535

// GFX90A: ds_dec_u64 v1, a[2:3]           ; encoding: [0x00,0x00,0x88,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_dec_u64 v1, a[2:3]

// GFX90A: ds_dec_u64 v1, a[2:3]           ; encoding: [0x00,0x00,0x88,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_dec_u64 v1, a[2:3]

// GFX90A: ds_dec_u64 v1, a[2:3] offset:4  ; encoding: [0x04,0x00,0x88,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_dec_u64 v1, a[2:3] offset:4

// GFX90A: ds_min_i64 v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x8a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_i64 v1, a[2:3] offset:65535

// GFX90A: ds_min_i64 v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x8a,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_i64 v255, a[2:3] offset:65535

// GFX90A: ds_min_i64 v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0x8a,0xda,0x01,0xfe,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_i64 v1, a[254:255] offset:65535

// GFX90A: ds_min_i64 v1, a[2:3]           ; encoding: [0x00,0x00,0x8a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_i64 v1, a[2:3]

// GFX90A: ds_min_i64 v1, a[2:3]           ; encoding: [0x00,0x00,0x8a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_i64 v1, a[2:3]

// GFX90A: ds_min_i64 v1, a[2:3] offset:4  ; encoding: [0x04,0x00,0x8a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_i64 v1, a[2:3] offset:4

// GFX90A: ds_max_i64 v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x8c,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_i64 v1, a[2:3] offset:65535

// GFX90A: ds_max_i64 v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x8c,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_i64 v255, a[2:3] offset:65535

// GFX90A: ds_max_i64 v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0x8c,0xda,0x01,0xfe,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_i64 v1, a[254:255] offset:65535

// GFX90A: ds_max_i64 v1, a[2:3]           ; encoding: [0x00,0x00,0x8c,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_i64 v1, a[2:3]

// GFX90A: ds_max_i64 v1, a[2:3]           ; encoding: [0x00,0x00,0x8c,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_i64 v1, a[2:3]

// GFX90A: ds_max_i64 v1, a[2:3] offset:4  ; encoding: [0x04,0x00,0x8c,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_i64 v1, a[2:3] offset:4

// GFX90A: ds_min_u64 v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x8e,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_u64 v1, a[2:3] offset:65535

// GFX90A: ds_min_u64 v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x8e,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_u64 v255, a[2:3] offset:65535

// GFX90A: ds_min_u64 v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0x8e,0xda,0x01,0xfe,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_u64 v1, a[254:255] offset:65535

// GFX90A: ds_min_u64 v1, a[2:3]           ; encoding: [0x00,0x00,0x8e,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_u64 v1, a[2:3]

// GFX90A: ds_min_u64 v1, a[2:3]           ; encoding: [0x00,0x00,0x8e,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_u64 v1, a[2:3]

// GFX90A: ds_min_u64 v1, a[2:3] offset:4  ; encoding: [0x04,0x00,0x8e,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_u64 v1, a[2:3] offset:4

// GFX90A: ds_max_u64 v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x90,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_u64 v1, a[2:3] offset:65535

// GFX90A: ds_max_u64 v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x90,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_u64 v255, a[2:3] offset:65535

// GFX90A: ds_max_u64 v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0x90,0xda,0x01,0xfe,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_u64 v1, a[254:255] offset:65535

// GFX90A: ds_max_u64 v1, a[2:3]           ; encoding: [0x00,0x00,0x90,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_u64 v1, a[2:3]

// GFX90A: ds_max_u64 v1, a[2:3]           ; encoding: [0x00,0x00,0x90,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_u64 v1, a[2:3]

// GFX90A: ds_max_u64 v1, a[2:3] offset:4  ; encoding: [0x04,0x00,0x90,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_u64 v1, a[2:3] offset:4

// GFX90A: ds_and_b64 v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x92,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_and_b64 v1, a[2:3] offset:65535

// GFX90A: ds_and_b64 v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x92,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_and_b64 v255, a[2:3] offset:65535

// GFX90A: ds_and_b64 v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0x92,0xda,0x01,0xfe,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_and_b64 v1, a[254:255] offset:65535

// GFX90A: ds_and_b64 v1, a[2:3]           ; encoding: [0x00,0x00,0x92,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_and_b64 v1, a[2:3]

// GFX90A: ds_and_b64 v1, a[2:3]           ; encoding: [0x00,0x00,0x92,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_and_b64 v1, a[2:3]

// GFX90A: ds_and_b64 v1, a[2:3] offset:4  ; encoding: [0x04,0x00,0x92,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_and_b64 v1, a[2:3] offset:4

// GFX90A: ds_or_b64 v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x94,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_or_b64 v1, a[2:3] offset:65535

// GFX90A: ds_or_b64 v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x94,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_or_b64 v255, a[2:3] offset:65535

// GFX90A: ds_or_b64 v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0x94,0xda,0x01,0xfe,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_or_b64 v1, a[254:255] offset:65535

// GFX90A: ds_or_b64 v1, a[2:3]            ; encoding: [0x00,0x00,0x94,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_or_b64 v1, a[2:3]

// GFX90A: ds_or_b64 v1, a[2:3]            ; encoding: [0x00,0x00,0x94,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_or_b64 v1, a[2:3]

// GFX90A: ds_or_b64 v1, a[2:3] offset:4   ; encoding: [0x04,0x00,0x94,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_or_b64 v1, a[2:3] offset:4

// GFX90A: ds_xor_b64 v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x96,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_xor_b64 v1, a[2:3] offset:65535

// GFX90A: ds_xor_b64 v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x96,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_xor_b64 v255, a[2:3] offset:65535

// GFX90A: ds_xor_b64 v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0x96,0xda,0x01,0xfe,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_xor_b64 v1, a[254:255] offset:65535

// GFX90A: ds_xor_b64 v1, a[2:3]           ; encoding: [0x00,0x00,0x96,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_xor_b64 v1, a[2:3]

// GFX90A: ds_xor_b64 v1, a[2:3]           ; encoding: [0x00,0x00,0x96,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_xor_b64 v1, a[2:3]

// GFX90A: ds_xor_b64 v1, a[2:3] offset:4  ; encoding: [0x04,0x00,0x96,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_xor_b64 v1, a[2:3] offset:4

// GFX90A: ds_mskor_b64 v1, a[2:3], a[4:5] offset:65535 ; encoding: [0xff,0xff,0x98,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_b64 v1, a[2:3], a[4:5] offset:65535

// GFX90A: ds_mskor_b64 v255, a[2:3], a[4:5] offset:65535 ; encoding: [0xff,0xff,0x98,0xda,0xff,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_b64 v255, a[2:3], a[4:5] offset:65535

// GFX90A: ds_mskor_b64 v1, a[254:255], a[4:5] offset:65535 ; encoding: [0xff,0xff,0x98,0xda,0x01,0xfe,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_b64 v1, a[254:255], a[4:5] offset:65535

// GFX90A: ds_mskor_b64 v1, a[2:3], a[254:255] offset:65535 ; encoding: [0xff,0xff,0x98,0xda,0x01,0x02,0xfe,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_b64 v1, a[2:3], a[254:255] offset:65535

// GFX90A: ds_mskor_b64 v1, a[2:3], a[4:5] ; encoding: [0x00,0x00,0x98,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_b64 v1, a[2:3], a[4:5]

// GFX90A: ds_mskor_b64 v1, a[2:3], a[4:5] ; encoding: [0x00,0x00,0x98,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_b64 v1, a[2:3], a[4:5]

// GFX90A: ds_mskor_b64 v1, a[2:3], a[4:5] offset:4 ; encoding: [0x04,0x00,0x98,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_b64 v1, a[2:3], a[4:5] offset:4

// GFX90A: ds_write_b64 v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x9a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b64 v1, a[2:3] offset:65535

// GFX90A: ds_write_b64 v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0x9a,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b64 v255, a[2:3] offset:65535

// GFX90A: ds_write_b64 v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0x9a,0xda,0x01,0xfe,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b64 v1, a[254:255] offset:65535

// GFX90A: ds_write_b64 v1, a[2:3]         ; encoding: [0x00,0x00,0x9a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b64 v1, a[2:3]

// GFX90A: ds_write_b64 v1, a[2:3]         ; encoding: [0x00,0x00,0x9a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b64 v1, a[2:3]

// GFX90A: ds_write_b64 v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0x9a,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b64 v1, a[2:3] offset:4

// GFX90A: ds_write2_b64 v1, a[2:3], a[4:5] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x9c,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b64 v1, a[2:3], a[4:5] offset0:127 offset1:255

// GFX90A: ds_write2_b64 v255, a[2:3], a[4:5] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x9c,0xda,0xff,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b64 v255, a[2:3], a[4:5] offset0:127 offset1:255

// GFX90A: ds_write2_b64 v1, a[254:255], a[4:5] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x9c,0xda,0x01,0xfe,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b64 v1, a[254:255], a[4:5] offset0:127 offset1:255

// GFX90A: ds_write2_b64 v1, a[2:3], a[254:255] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x9c,0xda,0x01,0x02,0xfe,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b64 v1, a[2:3], a[254:255] offset0:127 offset1:255

// GFX90A: ds_write2_b64 v1, a[2:3], a[4:5] offset1:255 ; encoding: [0x00,0xff,0x9c,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b64 v1, a[2:3], a[4:5] offset1:255

// GFX90A: ds_write2_b64 v1, a[2:3], a[4:5] offset1:255 ; encoding: [0x00,0xff,0x9c,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b64 v1, a[2:3], a[4:5] offset1:255

// GFX90A: ds_write2_b64 v1, a[2:3], a[4:5] offset0:16 offset1:255 ; encoding: [0x10,0xff,0x9c,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b64 v1, a[2:3], a[4:5] offset0:16 offset1:255

// GFX90A: ds_write2_b64 v1, a[2:3], a[4:5] offset0:127 ; encoding: [0x7f,0x00,0x9c,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b64 v1, a[2:3], a[4:5] offset0:127

// GFX90A: ds_write2_b64 v1, a[2:3], a[4:5] offset0:127 ; encoding: [0x7f,0x00,0x9c,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b64 v1, a[2:3], a[4:5] offset0:127

// GFX90A: ds_write2_b64 v1, a[2:3], a[4:5] offset0:127 offset1:1 ; encoding: [0x7f,0x01,0x9c,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2_b64 v1, a[2:3], a[4:5] offset0:127 offset1:1

// GFX90A: ds_write2st64_b64 v1, a[2:3], a[4:5] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x9e,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b64 v1, a[2:3], a[4:5] offset0:127 offset1:255

// GFX90A: ds_write2st64_b64 v255, a[2:3], a[4:5] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x9e,0xda,0xff,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b64 v255, a[2:3], a[4:5] offset0:127 offset1:255

// GFX90A: ds_write2st64_b64 v1, a[254:255], a[4:5] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x9e,0xda,0x01,0xfe,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b64 v1, a[254:255], a[4:5] offset0:127 offset1:255

// GFX90A: ds_write2st64_b64 v1, a[2:3], a[254:255] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0x9e,0xda,0x01,0x02,0xfe,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b64 v1, a[2:3], a[254:255] offset0:127 offset1:255

// GFX90A: ds_write2st64_b64 v1, a[2:3], a[4:5] offset1:255 ; encoding: [0x00,0xff,0x9e,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b64 v1, a[2:3], a[4:5] offset1:255

// GFX90A: ds_write2st64_b64 v1, a[2:3], a[4:5] offset1:255 ; encoding: [0x00,0xff,0x9e,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b64 v1, a[2:3], a[4:5] offset1:255

// GFX90A: ds_write2st64_b64 v1, a[2:3], a[4:5] offset0:16 offset1:255 ; encoding: [0x10,0xff,0x9e,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b64 v1, a[2:3], a[4:5] offset0:16 offset1:255

// GFX90A: ds_write2st64_b64 v1, a[2:3], a[4:5] offset0:127 ; encoding: [0x7f,0x00,0x9e,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b64 v1, a[2:3], a[4:5] offset0:127

// GFX90A: ds_write2st64_b64 v1, a[2:3], a[4:5] offset0:127 ; encoding: [0x7f,0x00,0x9e,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b64 v1, a[2:3], a[4:5] offset0:127

// GFX90A: ds_write2st64_b64 v1, a[2:3], a[4:5] offset0:127 offset1:1 ; encoding: [0x7f,0x01,0x9e,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_write2st64_b64 v1, a[2:3], a[4:5] offset0:127 offset1:1

// GFX90A: ds_cmpst_b64 v1, a[2:3], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xa0,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_b64 v1, a[2:3], a[4:5] offset:65535

// GFX90A: ds_cmpst_b64 v255, a[2:3], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xa0,0xda,0xff,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_b64 v255, a[2:3], a[4:5] offset:65535

// GFX90A: ds_cmpst_b64 v1, a[254:255], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xa0,0xda,0x01,0xfe,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_b64 v1, a[254:255], a[4:5] offset:65535

// GFX90A: ds_cmpst_b64 v1, a[2:3], a[254:255] offset:65535 ; encoding: [0xff,0xff,0xa0,0xda,0x01,0x02,0xfe,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_b64 v1, a[2:3], a[254:255] offset:65535

// GFX90A: ds_cmpst_b64 v1, a[2:3], a[4:5] ; encoding: [0x00,0x00,0xa0,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_b64 v1, a[2:3], a[4:5]

// GFX90A: ds_cmpst_b64 v1, a[2:3], a[4:5] ; encoding: [0x00,0x00,0xa0,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_b64 v1, a[2:3], a[4:5]

// GFX90A: ds_cmpst_b64 v1, a[2:3], a[4:5] offset:4 ; encoding: [0x04,0x00,0xa0,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_b64 v1, a[2:3], a[4:5] offset:4

// GFX90A: ds_cmpst_f64 v1, a[2:3], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xa2,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_f64 v1, a[2:3], a[4:5] offset:65535

// GFX90A: ds_cmpst_f64 v255, a[2:3], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xa2,0xda,0xff,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_f64 v255, a[2:3], a[4:5] offset:65535

// GFX90A: ds_cmpst_f64 v1, a[254:255], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xa2,0xda,0x01,0xfe,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_f64 v1, a[254:255], a[4:5] offset:65535

// GFX90A: ds_cmpst_f64 v1, a[2:3], a[254:255] offset:65535 ; encoding: [0xff,0xff,0xa2,0xda,0x01,0x02,0xfe,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_f64 v1, a[2:3], a[254:255] offset:65535

// GFX90A: ds_cmpst_f64 v1, a[2:3], a[4:5] ; encoding: [0x00,0x00,0xa2,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_f64 v1, a[2:3], a[4:5]

// GFX90A: ds_cmpst_f64 v1, a[2:3], a[4:5] ; encoding: [0x00,0x00,0xa2,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_f64 v1, a[2:3], a[4:5]

// GFX90A: ds_cmpst_f64 v1, a[2:3], a[4:5] offset:4 ; encoding: [0x04,0x00,0xa2,0xda,0x01,0x02,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_f64 v1, a[2:3], a[4:5] offset:4

// GFX90A: ds_min_f64 v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xa4,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_f64 v1, a[2:3] offset:65535

// GFX90A: ds_min_f64 v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xa4,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_f64 v255, a[2:3] offset:65535

// GFX90A: ds_min_f64 v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xa4,0xda,0x01,0xfe,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_f64 v1, a[254:255] offset:65535

// GFX90A: ds_min_f64 v1, a[2:3]           ; encoding: [0x00,0x00,0xa4,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_f64 v1, a[2:3]

// GFX90A: ds_min_f64 v1, a[2:3]           ; encoding: [0x00,0x00,0xa4,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_f64 v1, a[2:3]

// GFX90A: ds_min_f64 v1, a[2:3] offset:4  ; encoding: [0x04,0x00,0xa4,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_min_f64 v1, a[2:3] offset:4

// GFX90A: ds_max_f64 v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xa6,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_f64 v1, a[2:3] offset:65535

// GFX90A: ds_max_f64 v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xa6,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_f64 v255, a[2:3] offset:65535

// GFX90A: ds_max_f64 v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xa6,0xda,0x01,0xfe,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_f64 v1, a[254:255] offset:65535

// GFX90A: ds_max_f64 v1, a[2:3]           ; encoding: [0x00,0x00,0xa6,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_f64 v1, a[2:3]

// GFX90A: ds_max_f64 v1, a[2:3]           ; encoding: [0x00,0x00,0xa6,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_f64 v1, a[2:3]

// GFX90A: ds_max_f64 v1, a[2:3] offset:4  ; encoding: [0x04,0x00,0xa6,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_max_f64 v1, a[2:3] offset:4

// GFX90A: ds_write_b8_d16_hi v1, a2 offset:65535 ; encoding: [0xff,0xff,0xa8,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b8_d16_hi v1, a2 offset:65535

// GFX90A: ds_write_b8_d16_hi v255, a2 offset:65535 ; encoding: [0xff,0xff,0xa8,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b8_d16_hi v255, a2 offset:65535

// GFX90A: ds_write_b8_d16_hi v1, a255 offset:65535 ; encoding: [0xff,0xff,0xa8,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b8_d16_hi v1, a255 offset:65535

// GFX90A: ds_write_b8_d16_hi v1, a2       ; encoding: [0x00,0x00,0xa8,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b8_d16_hi v1, a2

// GFX90A: ds_write_b8_d16_hi v1, a2       ; encoding: [0x00,0x00,0xa8,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b8_d16_hi v1, a2

// GFX90A: ds_write_b8_d16_hi v1, a2 offset:4 ; encoding: [0x04,0x00,0xa8,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b8_d16_hi v1, a2 offset:4

// GFX90A: ds_write_b16_d16_hi v1, a2 offset:65535 ; encoding: [0xff,0xff,0xaa,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b16_d16_hi v1, a2 offset:65535

// GFX90A: ds_write_b16_d16_hi v255, a2 offset:65535 ; encoding: [0xff,0xff,0xaa,0xda,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b16_d16_hi v255, a2 offset:65535

// GFX90A: ds_write_b16_d16_hi v1, a255 offset:65535 ; encoding: [0xff,0xff,0xaa,0xda,0x01,0xff,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b16_d16_hi v1, a255 offset:65535

// GFX90A: ds_write_b16_d16_hi v1, a2      ; encoding: [0x00,0x00,0xaa,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b16_d16_hi v1, a2

// GFX90A: ds_write_b16_d16_hi v1, a2      ; encoding: [0x00,0x00,0xaa,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b16_d16_hi v1, a2

// GFX90A: ds_write_b16_d16_hi v1, a2 offset:4 ; encoding: [0x04,0x00,0xaa,0xda,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b16_d16_hi v1, a2 offset:4

// GFX90A: ds_read_u8_d16 a5, v1 offset:65535 ; encoding: [0xff,0xff,0xac,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8_d16 a5, v1 offset:65535

// GFX90A: ds_read_u8_d16 a255, v1 offset:65535 ; encoding: [0xff,0xff,0xac,0xda,0x01,0x00,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8_d16 a255, v1 offset:65535

// GFX90A: ds_read_u8_d16 a5, v255 offset:65535 ; encoding: [0xff,0xff,0xac,0xda,0xff,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8_d16 a5, v255 offset:65535

// GFX90A: ds_read_u8_d16 a5, v1           ; encoding: [0x00,0x00,0xac,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8_d16 a5, v1

// GFX90A: ds_read_u8_d16 a5, v1           ; encoding: [0x00,0x00,0xac,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8_d16 a5, v1

// GFX90A: ds_read_u8_d16 a5, v1 offset:4  ; encoding: [0x04,0x00,0xac,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8_d16 a5, v1 offset:4

// GFX90A: ds_read_u8_d16_hi a5, v1 offset:65535 ; encoding: [0xff,0xff,0xae,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8_d16_hi a5, v1 offset:65535

// GFX90A: ds_read_u8_d16_hi a255, v1 offset:65535 ; encoding: [0xff,0xff,0xae,0xda,0x01,0x00,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8_d16_hi a255, v1 offset:65535

// GFX90A: ds_read_u8_d16_hi a5, v255 offset:65535 ; encoding: [0xff,0xff,0xae,0xda,0xff,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8_d16_hi a5, v255 offset:65535

// GFX90A: ds_read_u8_d16_hi a5, v1        ; encoding: [0x00,0x00,0xae,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8_d16_hi a5, v1

// GFX90A: ds_read_u8_d16_hi a5, v1        ; encoding: [0x00,0x00,0xae,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8_d16_hi a5, v1

// GFX90A: ds_read_u8_d16_hi a5, v1 offset:4 ; encoding: [0x04,0x00,0xae,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u8_d16_hi a5, v1 offset:4

// GFX90A: ds_read_i8_d16 a5, v1 offset:65535 ; encoding: [0xff,0xff,0xb0,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8_d16 a5, v1 offset:65535

// GFX90A: ds_read_i8_d16 a255, v1 offset:65535 ; encoding: [0xff,0xff,0xb0,0xda,0x01,0x00,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8_d16 a255, v1 offset:65535

// GFX90A: ds_read_i8_d16 a5, v255 offset:65535 ; encoding: [0xff,0xff,0xb0,0xda,0xff,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8_d16 a5, v255 offset:65535

// GFX90A: ds_read_i8_d16 a5, v1           ; encoding: [0x00,0x00,0xb0,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8_d16 a5, v1

// GFX90A: ds_read_i8_d16 a5, v1           ; encoding: [0x00,0x00,0xb0,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8_d16 a5, v1

// GFX90A: ds_read_i8_d16 a5, v1 offset:4  ; encoding: [0x04,0x00,0xb0,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8_d16 a5, v1 offset:4

// GFX90A: ds_read_i8_d16_hi a5, v1 offset:65535 ; encoding: [0xff,0xff,0xb2,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8_d16_hi a5, v1 offset:65535

// GFX90A: ds_read_i8_d16_hi a255, v1 offset:65535 ; encoding: [0xff,0xff,0xb2,0xda,0x01,0x00,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8_d16_hi a255, v1 offset:65535

// GFX90A: ds_read_i8_d16_hi a5, v255 offset:65535 ; encoding: [0xff,0xff,0xb2,0xda,0xff,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8_d16_hi a5, v255 offset:65535

// GFX90A: ds_read_i8_d16_hi a5, v1        ; encoding: [0x00,0x00,0xb2,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8_d16_hi a5, v1

// GFX90A: ds_read_i8_d16_hi a5, v1        ; encoding: [0x00,0x00,0xb2,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8_d16_hi a5, v1

// GFX90A: ds_read_i8_d16_hi a5, v1 offset:4 ; encoding: [0x04,0x00,0xb2,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_i8_d16_hi a5, v1 offset:4

// GFX90A: ds_read_u16_d16 a5, v1 offset:65535 ; encoding: [0xff,0xff,0xb4,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16_d16 a5, v1 offset:65535

// GFX90A: ds_read_u16_d16 a255, v1 offset:65535 ; encoding: [0xff,0xff,0xb4,0xda,0x01,0x00,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16_d16 a255, v1 offset:65535

// GFX90A: ds_read_u16_d16 a5, v255 offset:65535 ; encoding: [0xff,0xff,0xb4,0xda,0xff,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16_d16 a5, v255 offset:65535

// GFX90A: ds_read_u16_d16 a5, v1          ; encoding: [0x00,0x00,0xb4,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16_d16 a5, v1

// GFX90A: ds_read_u16_d16 a5, v1          ; encoding: [0x00,0x00,0xb4,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16_d16 a5, v1

// GFX90A: ds_read_u16_d16 a5, v1 offset:4 ; encoding: [0x04,0x00,0xb4,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16_d16 a5, v1 offset:4

// GFX90A: ds_read_u16_d16_hi a5, v1 offset:65535 ; encoding: [0xff,0xff,0xb6,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16_d16_hi a5, v1 offset:65535

// GFX90A: ds_read_u16_d16_hi a255, v1 offset:65535 ; encoding: [0xff,0xff,0xb6,0xda,0x01,0x00,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16_d16_hi a255, v1 offset:65535

// GFX90A: ds_read_u16_d16_hi a5, v255 offset:65535 ; encoding: [0xff,0xff,0xb6,0xda,0xff,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16_d16_hi a5, v255 offset:65535

// GFX90A: ds_read_u16_d16_hi a5, v1       ; encoding: [0x00,0x00,0xb6,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16_d16_hi a5, v1

// GFX90A: ds_read_u16_d16_hi a5, v1       ; encoding: [0x00,0x00,0xb6,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16_d16_hi a5, v1

// GFX90A: ds_read_u16_d16_hi a5, v1 offset:4 ; encoding: [0x04,0x00,0xb6,0xda,0x01,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_u16_d16_hi a5, v1 offset:4

// GFX90A: ds_add_rtn_u64 a[6:7], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xc0,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_u64 a[6:7], v1, a[2:3] offset:65535

// GFX90A: ds_add_rtn_u64 a[254:255], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xc0,0xda,0x01,0x02,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_u64 a[254:255], v1, a[2:3] offset:65535

// GFX90A: ds_add_rtn_u64 a[6:7], v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xc0,0xda,0xff,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_u64 a[6:7], v255, a[2:3] offset:65535

// GFX90A: ds_add_rtn_u64 a[6:7], v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xc0,0xda,0x01,0xfe,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_u64 a[6:7], v1, a[254:255] offset:65535

// GFX90A: ds_add_rtn_u64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xc0,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_u64 a[6:7], v1, a[2:3]

// GFX90A: ds_add_rtn_u64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xc0,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_u64 a[6:7], v1, a[2:3]

// GFX90A: ds_add_rtn_u64 a[6:7], v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0xc0,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_add_rtn_u64 a[6:7], v1, a[2:3] offset:4

// GFX90A: ds_sub_rtn_u64 a[6:7], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xc2,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_sub_rtn_u64 a[6:7], v1, a[2:3] offset:65535

// GFX90A: ds_sub_rtn_u64 a[254:255], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xc2,0xda,0x01,0x02,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_sub_rtn_u64 a[254:255], v1, a[2:3] offset:65535

// GFX90A: ds_sub_rtn_u64 a[6:7], v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xc2,0xda,0xff,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_sub_rtn_u64 a[6:7], v255, a[2:3] offset:65535

// GFX90A: ds_sub_rtn_u64 a[6:7], v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xc2,0xda,0x01,0xfe,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_sub_rtn_u64 a[6:7], v1, a[254:255] offset:65535

// GFX90A: ds_sub_rtn_u64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xc2,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_sub_rtn_u64 a[6:7], v1, a[2:3]

// GFX90A: ds_sub_rtn_u64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xc2,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_sub_rtn_u64 a[6:7], v1, a[2:3]

// GFX90A: ds_sub_rtn_u64 a[6:7], v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0xc2,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_sub_rtn_u64 a[6:7], v1, a[2:3] offset:4

// GFX90A: ds_rsub_rtn_u64 a[6:7], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xc4,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_rsub_rtn_u64 a[6:7], v1, a[2:3] offset:65535

// GFX90A: ds_rsub_rtn_u64 a[254:255], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xc4,0xda,0x01,0x02,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_rsub_rtn_u64 a[254:255], v1, a[2:3] offset:65535

// GFX90A: ds_rsub_rtn_u64 a[6:7], v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xc4,0xda,0xff,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_rsub_rtn_u64 a[6:7], v255, a[2:3] offset:65535

// GFX90A: ds_rsub_rtn_u64 a[6:7], v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xc4,0xda,0x01,0xfe,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_rsub_rtn_u64 a[6:7], v1, a[254:255] offset:65535

// GFX90A: ds_rsub_rtn_u64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xc4,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_rsub_rtn_u64 a[6:7], v1, a[2:3]

// GFX90A: ds_rsub_rtn_u64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xc4,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_rsub_rtn_u64 a[6:7], v1, a[2:3]

// GFX90A: ds_rsub_rtn_u64 a[6:7], v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0xc4,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_rsub_rtn_u64 a[6:7], v1, a[2:3] offset:4

// GFX90A: ds_inc_rtn_u64 a[6:7], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xc6,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_inc_rtn_u64 a[6:7], v1, a[2:3] offset:65535

// GFX90A: ds_inc_rtn_u64 a[254:255], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xc6,0xda,0x01,0x02,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_inc_rtn_u64 a[254:255], v1, a[2:3] offset:65535

// GFX90A: ds_inc_rtn_u64 a[6:7], v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xc6,0xda,0xff,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_inc_rtn_u64 a[6:7], v255, a[2:3] offset:65535

// GFX90A: ds_inc_rtn_u64 a[6:7], v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xc6,0xda,0x01,0xfe,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_inc_rtn_u64 a[6:7], v1, a[254:255] offset:65535

// GFX90A: ds_inc_rtn_u64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xc6,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_inc_rtn_u64 a[6:7], v1, a[2:3]

// GFX90A: ds_inc_rtn_u64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xc6,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_inc_rtn_u64 a[6:7], v1, a[2:3]

// GFX90A: ds_inc_rtn_u64 a[6:7], v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0xc6,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_inc_rtn_u64 a[6:7], v1, a[2:3] offset:4

// GFX90A: ds_dec_rtn_u64 a[6:7], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xc8,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_dec_rtn_u64 a[6:7], v1, a[2:3] offset:65535

// GFX90A: ds_dec_rtn_u64 a[254:255], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xc8,0xda,0x01,0x02,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_dec_rtn_u64 a[254:255], v1, a[2:3] offset:65535

// GFX90A: ds_dec_rtn_u64 a[6:7], v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xc8,0xda,0xff,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_dec_rtn_u64 a[6:7], v255, a[2:3] offset:65535

// GFX90A: ds_dec_rtn_u64 a[6:7], v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xc8,0xda,0x01,0xfe,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_dec_rtn_u64 a[6:7], v1, a[254:255] offset:65535

// GFX90A: ds_dec_rtn_u64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xc8,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_dec_rtn_u64 a[6:7], v1, a[2:3]

// GFX90A: ds_dec_rtn_u64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xc8,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_dec_rtn_u64 a[6:7], v1, a[2:3]

// GFX90A: ds_dec_rtn_u64 a[6:7], v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0xc8,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_dec_rtn_u64 a[6:7], v1, a[2:3] offset:4

// GFX90A: ds_min_rtn_i64 a[6:7], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xca,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_i64 a[6:7], v1, a[2:3] offset:65535

// GFX90A: ds_min_rtn_i64 a[254:255], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xca,0xda,0x01,0x02,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_i64 a[254:255], v1, a[2:3] offset:65535

// GFX90A: ds_min_rtn_i64 a[6:7], v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xca,0xda,0xff,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_i64 a[6:7], v255, a[2:3] offset:65535

// GFX90A: ds_min_rtn_i64 a[6:7], v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xca,0xda,0x01,0xfe,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_i64 a[6:7], v1, a[254:255] offset:65535

// GFX90A: ds_min_rtn_i64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xca,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_i64 a[6:7], v1, a[2:3]

// GFX90A: ds_min_rtn_i64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xca,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_i64 a[6:7], v1, a[2:3]

// GFX90A: ds_min_rtn_i64 a[6:7], v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0xca,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_i64 a[6:7], v1, a[2:3] offset:4

// GFX90A: ds_max_rtn_i64 a[6:7], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xcc,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_i64 a[6:7], v1, a[2:3] offset:65535

// GFX90A: ds_max_rtn_i64 a[254:255], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xcc,0xda,0x01,0x02,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_i64 a[254:255], v1, a[2:3] offset:65535

// GFX90A: ds_max_rtn_i64 a[6:7], v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xcc,0xda,0xff,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_i64 a[6:7], v255, a[2:3] offset:65535

// GFX90A: ds_max_rtn_i64 a[6:7], v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xcc,0xda,0x01,0xfe,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_i64 a[6:7], v1, a[254:255] offset:65535

// GFX90A: ds_max_rtn_i64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xcc,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_i64 a[6:7], v1, a[2:3]

// GFX90A: ds_max_rtn_i64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xcc,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_i64 a[6:7], v1, a[2:3]

// GFX90A: ds_max_rtn_i64 a[6:7], v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0xcc,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_i64 a[6:7], v1, a[2:3] offset:4

// GFX90A: ds_min_rtn_u64 a[6:7], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xce,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_u64 a[6:7], v1, a[2:3] offset:65535

// GFX90A: ds_min_rtn_u64 a[254:255], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xce,0xda,0x01,0x02,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_u64 a[254:255], v1, a[2:3] offset:65535

// GFX90A: ds_min_rtn_u64 a[6:7], v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xce,0xda,0xff,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_u64 a[6:7], v255, a[2:3] offset:65535

// GFX90A: ds_min_rtn_u64 a[6:7], v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xce,0xda,0x01,0xfe,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_u64 a[6:7], v1, a[254:255] offset:65535

// GFX90A: ds_min_rtn_u64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xce,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_u64 a[6:7], v1, a[2:3]

// GFX90A: ds_min_rtn_u64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xce,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_u64 a[6:7], v1, a[2:3]

// GFX90A: ds_min_rtn_u64 a[6:7], v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0xce,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_u64 a[6:7], v1, a[2:3] offset:4

// GFX90A: ds_max_rtn_u64 a[6:7], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xd0,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_u64 a[6:7], v1, a[2:3] offset:65535

// GFX90A: ds_max_rtn_u64 a[254:255], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xd0,0xda,0x01,0x02,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_u64 a[254:255], v1, a[2:3] offset:65535

// GFX90A: ds_max_rtn_u64 a[6:7], v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xd0,0xda,0xff,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_u64 a[6:7], v255, a[2:3] offset:65535

// GFX90A: ds_max_rtn_u64 a[6:7], v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xd0,0xda,0x01,0xfe,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_u64 a[6:7], v1, a[254:255] offset:65535

// GFX90A: ds_max_rtn_u64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xd0,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_u64 a[6:7], v1, a[2:3]

// GFX90A: ds_max_rtn_u64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xd0,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_u64 a[6:7], v1, a[2:3]

// GFX90A: ds_max_rtn_u64 a[6:7], v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0xd0,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_u64 a[6:7], v1, a[2:3] offset:4

// GFX90A: ds_and_rtn_b64 a[6:7], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xd2,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_and_rtn_b64 a[6:7], v1, a[2:3] offset:65535

// GFX90A: ds_and_rtn_b64 a[254:255], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xd2,0xda,0x01,0x02,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_and_rtn_b64 a[254:255], v1, a[2:3] offset:65535

// GFX90A: ds_and_rtn_b64 a[6:7], v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xd2,0xda,0xff,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_and_rtn_b64 a[6:7], v255, a[2:3] offset:65535

// GFX90A: ds_and_rtn_b64 a[6:7], v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xd2,0xda,0x01,0xfe,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_and_rtn_b64 a[6:7], v1, a[254:255] offset:65535

// GFX90A: ds_and_rtn_b64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xd2,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_and_rtn_b64 a[6:7], v1, a[2:3]

// GFX90A: ds_and_rtn_b64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xd2,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_and_rtn_b64 a[6:7], v1, a[2:3]

// GFX90A: ds_and_rtn_b64 a[6:7], v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0xd2,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_and_rtn_b64 a[6:7], v1, a[2:3] offset:4

// GFX90A: ds_or_rtn_b64 a[6:7], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xd4,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_or_rtn_b64 a[6:7], v1, a[2:3] offset:65535

// GFX90A: ds_or_rtn_b64 a[254:255], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xd4,0xda,0x01,0x02,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_or_rtn_b64 a[254:255], v1, a[2:3] offset:65535

// GFX90A: ds_or_rtn_b64 a[6:7], v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xd4,0xda,0xff,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_or_rtn_b64 a[6:7], v255, a[2:3] offset:65535

// GFX90A: ds_or_rtn_b64 a[6:7], v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xd4,0xda,0x01,0xfe,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_or_rtn_b64 a[6:7], v1, a[254:255] offset:65535

// GFX90A: ds_or_rtn_b64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xd4,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_or_rtn_b64 a[6:7], v1, a[2:3]

// GFX90A: ds_or_rtn_b64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xd4,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_or_rtn_b64 a[6:7], v1, a[2:3]

// GFX90A: ds_or_rtn_b64 a[6:7], v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0xd4,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_or_rtn_b64 a[6:7], v1, a[2:3] offset:4

// GFX90A: ds_xor_rtn_b64 a[6:7], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xd6,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_xor_rtn_b64 a[6:7], v1, a[2:3] offset:65535

// GFX90A: ds_xor_rtn_b64 a[254:255], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xd6,0xda,0x01,0x02,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_xor_rtn_b64 a[254:255], v1, a[2:3] offset:65535

// GFX90A: ds_xor_rtn_b64 a[6:7], v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xd6,0xda,0xff,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_xor_rtn_b64 a[6:7], v255, a[2:3] offset:65535

// GFX90A: ds_xor_rtn_b64 a[6:7], v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xd6,0xda,0x01,0xfe,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_xor_rtn_b64 a[6:7], v1, a[254:255] offset:65535

// GFX90A: ds_xor_rtn_b64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xd6,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_xor_rtn_b64 a[6:7], v1, a[2:3]

// GFX90A: ds_xor_rtn_b64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xd6,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_xor_rtn_b64 a[6:7], v1, a[2:3]

// GFX90A: ds_xor_rtn_b64 a[6:7], v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0xd6,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_xor_rtn_b64 a[6:7], v1, a[2:3] offset:4

// GFX90A: ds_mskor_rtn_b64 a[6:7], v1, a[2:3], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xd8,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_rtn_b64 a[6:7], v1, a[2:3], a[4:5] offset:65535

// GFX90A: ds_mskor_rtn_b64 a[254:255], v1, a[2:3], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xd8,0xda,0x01,0x02,0x04,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_rtn_b64 a[254:255], v1, a[2:3], a[4:5] offset:65535

// GFX90A: ds_mskor_rtn_b64 a[6:7], v255, a[2:3], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xd8,0xda,0xff,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_rtn_b64 a[6:7], v255, a[2:3], a[4:5] offset:65535

// GFX90A: ds_mskor_rtn_b64 a[6:7], v1, a[254:255], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xd8,0xda,0x01,0xfe,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_rtn_b64 a[6:7], v1, a[254:255], a[4:5] offset:65535

// GFX90A: ds_mskor_rtn_b64 a[6:7], v1, a[2:3], a[254:255] offset:65535 ; encoding: [0xff,0xff,0xd8,0xda,0x01,0x02,0xfe,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_rtn_b64 a[6:7], v1, a[2:3], a[254:255] offset:65535

// GFX90A: ds_mskor_rtn_b64 a[6:7], v1, a[2:3], a[4:5] ; encoding: [0x00,0x00,0xd8,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_rtn_b64 a[6:7], v1, a[2:3], a[4:5]

// GFX90A: ds_mskor_rtn_b64 a[6:7], v1, a[2:3], a[4:5] ; encoding: [0x00,0x00,0xd8,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_rtn_b64 a[6:7], v1, a[2:3], a[4:5]

// GFX90A: ds_mskor_rtn_b64 a[6:7], v1, a[2:3], a[4:5] offset:4 ; encoding: [0x04,0x00,0xd8,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_mskor_rtn_b64 a[6:7], v1, a[2:3], a[4:5] offset:4

// GFX90A: ds_wrxchg_rtn_b64 a[6:7], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xda,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg_rtn_b64 a[6:7], v1, a[2:3] offset:65535

// GFX90A: ds_wrxchg_rtn_b64 a[254:255], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xda,0xda,0x01,0x02,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg_rtn_b64 a[254:255], v1, a[2:3] offset:65535

// GFX90A: ds_wrxchg_rtn_b64 a[6:7], v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xda,0xda,0xff,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg_rtn_b64 a[6:7], v255, a[2:3] offset:65535

// GFX90A: ds_wrxchg_rtn_b64 a[6:7], v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xda,0xda,0x01,0xfe,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg_rtn_b64 a[6:7], v1, a[254:255] offset:65535

// GFX90A: ds_wrxchg_rtn_b64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xda,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg_rtn_b64 a[6:7], v1, a[2:3]

// GFX90A: ds_wrxchg_rtn_b64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xda,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg_rtn_b64 a[6:7], v1, a[2:3]

// GFX90A: ds_wrxchg_rtn_b64 a[6:7], v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0xda,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg_rtn_b64 a[6:7], v1, a[2:3] offset:4

// GFX90A: ds_wrxchg2_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0xdc,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:127 offset1:255

// GFX90A: ds_wrxchg2_rtn_b64 a[252:255], v1, a[2:3], a[4:5] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0xdc,0xda,0x01,0x02,0x04,0xfc]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b64 a[252:255], v1, a[2:3], a[4:5] offset0:127 offset1:255

// GFX90A: ds_wrxchg2_rtn_b64 a[6:9], v255, a[2:3], a[4:5] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0xdc,0xda,0xff,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b64 a[6:9], v255, a[2:3], a[4:5] offset0:127 offset1:255

// GFX90A: ds_wrxchg2_rtn_b64 a[6:9], v1, a[254:255], a[4:5] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0xdc,0xda,0x01,0xfe,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b64 a[6:9], v1, a[254:255], a[4:5] offset0:127 offset1:255

// GFX90A: ds_wrxchg2_rtn_b64 a[6:9], v1, a[2:3], a[254:255] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0xdc,0xda,0x01,0x02,0xfe,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b64 a[6:9], v1, a[2:3], a[254:255] offset0:127 offset1:255

// GFX90A: ds_wrxchg2_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset1:255 ; encoding: [0x00,0xff,0xdc,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset1:255

// GFX90A: ds_wrxchg2_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset1:255 ; encoding: [0x00,0xff,0xdc,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset1:255

// GFX90A: ds_wrxchg2_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:16 offset1:255 ; encoding: [0x10,0xff,0xdc,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:16 offset1:255

// GFX90A: ds_wrxchg2_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:127 ; encoding: [0x7f,0x00,0xdc,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:127

// GFX90A: ds_wrxchg2_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:127 ; encoding: [0x7f,0x00,0xdc,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:127

// GFX90A: ds_wrxchg2_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:127 offset1:1 ; encoding: [0x7f,0x01,0xdc,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:127 offset1:1

// GFX90A: ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0xde,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:127 offset1:255

// GFX90A: ds_wrxchg2st64_rtn_b64 a[252:255], v1, a[2:3], a[4:5] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0xde,0xda,0x01,0x02,0x04,0xfc]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b64 a[252:255], v1, a[2:3], a[4:5] offset0:127 offset1:255

// GFX90A: ds_wrxchg2st64_rtn_b64 a[6:9], v255, a[2:3], a[4:5] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0xde,0xda,0xff,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b64 a[6:9], v255, a[2:3], a[4:5] offset0:127 offset1:255

// GFX90A: ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[254:255], a[4:5] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0xde,0xda,0x01,0xfe,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[254:255], a[4:5] offset0:127 offset1:255

// GFX90A: ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[2:3], a[254:255] offset0:127 offset1:255 ; encoding: [0x7f,0xff,0xde,0xda,0x01,0x02,0xfe,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[2:3], a[254:255] offset0:127 offset1:255

// GFX90A: ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset1:255 ; encoding: [0x00,0xff,0xde,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset1:255

// GFX90A: ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset1:255 ; encoding: [0x00,0xff,0xde,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset1:255

// GFX90A: ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:16 offset1:255 ; encoding: [0x10,0xff,0xde,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:16 offset1:255

// GFX90A: ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:127 ; encoding: [0x7f,0x00,0xde,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:127

// GFX90A: ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:127 ; encoding: [0x7f,0x00,0xde,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:127

// GFX90A: ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:127 offset1:1 ; encoding: [0x7f,0x01,0xde,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_wrxchg2st64_rtn_b64 a[6:9], v1, a[2:3], a[4:5] offset0:127 offset1:1

// GFX90A: ds_cmpst_rtn_b64 a[6:7], v1, a[2:3], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xe0,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_b64 a[6:7], v1, a[2:3], a[4:5] offset:65535

// GFX90A: ds_cmpst_rtn_b64 a[254:255], v1, a[2:3], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xe0,0xda,0x01,0x02,0x04,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_b64 a[254:255], v1, a[2:3], a[4:5] offset:65535

// GFX90A: ds_cmpst_rtn_b64 a[6:7], v255, a[2:3], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xe0,0xda,0xff,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_b64 a[6:7], v255, a[2:3], a[4:5] offset:65535

// GFX90A: ds_cmpst_rtn_b64 a[6:7], v1, a[254:255], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xe0,0xda,0x01,0xfe,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_b64 a[6:7], v1, a[254:255], a[4:5] offset:65535

// GFX90A: ds_cmpst_rtn_b64 a[6:7], v1, a[2:3], a[254:255] offset:65535 ; encoding: [0xff,0xff,0xe0,0xda,0x01,0x02,0xfe,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_b64 a[6:7], v1, a[2:3], a[254:255] offset:65535

// GFX90A: ds_cmpst_rtn_b64 a[6:7], v1, a[2:3], a[4:5] ; encoding: [0x00,0x00,0xe0,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_b64 a[6:7], v1, a[2:3], a[4:5]

// GFX90A: ds_cmpst_rtn_b64 a[6:7], v1, a[2:3], a[4:5] ; encoding: [0x00,0x00,0xe0,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_b64 a[6:7], v1, a[2:3], a[4:5]

// GFX90A: ds_cmpst_rtn_b64 a[6:7], v1, a[2:3], a[4:5] offset:4 ; encoding: [0x04,0x00,0xe0,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_b64 a[6:7], v1, a[2:3], a[4:5] offset:4

// GFX90A: ds_cmpst_rtn_f64 a[6:7], v1, a[2:3], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xe2,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_f64 a[6:7], v1, a[2:3], a[4:5] offset:65535

// GFX90A: ds_cmpst_rtn_f64 a[254:255], v1, a[2:3], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xe2,0xda,0x01,0x02,0x04,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_f64 a[254:255], v1, a[2:3], a[4:5] offset:65535

// GFX90A: ds_cmpst_rtn_f64 a[6:7], v255, a[2:3], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xe2,0xda,0xff,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_f64 a[6:7], v255, a[2:3], a[4:5] offset:65535

// GFX90A: ds_cmpst_rtn_f64 a[6:7], v1, a[254:255], a[4:5] offset:65535 ; encoding: [0xff,0xff,0xe2,0xda,0x01,0xfe,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_f64 a[6:7], v1, a[254:255], a[4:5] offset:65535

// GFX90A: ds_cmpst_rtn_f64 a[6:7], v1, a[2:3], a[254:255] offset:65535 ; encoding: [0xff,0xff,0xe2,0xda,0x01,0x02,0xfe,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_f64 a[6:7], v1, a[2:3], a[254:255] offset:65535

// GFX90A: ds_cmpst_rtn_f64 a[6:7], v1, a[2:3], a[4:5] ; encoding: [0x00,0x00,0xe2,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_f64 a[6:7], v1, a[2:3], a[4:5]

// GFX90A: ds_cmpst_rtn_f64 a[6:7], v1, a[2:3], a[4:5] ; encoding: [0x00,0x00,0xe2,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_f64 a[6:7], v1, a[2:3], a[4:5]

// GFX90A: ds_cmpst_rtn_f64 a[6:7], v1, a[2:3], a[4:5] offset:4 ; encoding: [0x04,0x00,0xe2,0xda,0x01,0x02,0x04,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_cmpst_rtn_f64 a[6:7], v1, a[2:3], a[4:5] offset:4

// GFX90A: ds_min_rtn_f64 a[6:7], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xe4,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_f64 a[6:7], v1, a[2:3] offset:65535

// GFX90A: ds_min_rtn_f64 a[254:255], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xe4,0xda,0x01,0x02,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_f64 a[254:255], v1, a[2:3] offset:65535

// GFX90A: ds_min_rtn_f64 a[6:7], v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xe4,0xda,0xff,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_f64 a[6:7], v255, a[2:3] offset:65535

// GFX90A: ds_min_rtn_f64 a[6:7], v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xe4,0xda,0x01,0xfe,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_f64 a[6:7], v1, a[254:255] offset:65535

// GFX90A: ds_min_rtn_f64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xe4,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_f64 a[6:7], v1, a[2:3]

// GFX90A: ds_min_rtn_f64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xe4,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_f64 a[6:7], v1, a[2:3]

// GFX90A: ds_min_rtn_f64 a[6:7], v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0xe4,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_min_rtn_f64 a[6:7], v1, a[2:3] offset:4

// GFX90A: ds_max_rtn_f64 a[6:7], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xe6,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_f64 a[6:7], v1, a[2:3] offset:65535

// GFX90A: ds_max_rtn_f64 a[254:255], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xe6,0xda,0x01,0x02,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_f64 a[254:255], v1, a[2:3] offset:65535

// GFX90A: ds_max_rtn_f64 a[6:7], v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xe6,0xda,0xff,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_f64 a[6:7], v255, a[2:3] offset:65535

// GFX90A: ds_max_rtn_f64 a[6:7], v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xe6,0xda,0x01,0xfe,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_f64 a[6:7], v1, a[254:255] offset:65535

// GFX90A: ds_max_rtn_f64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xe6,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_f64 a[6:7], v1, a[2:3]

// GFX90A: ds_max_rtn_f64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xe6,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_f64 a[6:7], v1, a[2:3]

// GFX90A: ds_max_rtn_f64 a[6:7], v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0xe6,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_max_rtn_f64 a[6:7], v1, a[2:3] offset:4

// GFX90A: ds_read_b64 a[6:7], v1 offset:65535 ; encoding: [0xff,0xff,0xec,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b64 a[6:7], v1 offset:65535

// GFX90A: ds_read_b64 a[254:255], v1 offset:65535 ; encoding: [0xff,0xff,0xec,0xda,0x01,0x00,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b64 a[254:255], v1 offset:65535

// GFX90A: ds_read_b64 a[6:7], v255 offset:65535 ; encoding: [0xff,0xff,0xec,0xda,0xff,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b64 a[6:7], v255 offset:65535

// GFX90A: ds_read_b64 a[6:7], v1          ; encoding: [0x00,0x00,0xec,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b64 a[6:7], v1

// GFX90A: ds_read_b64 a[6:7], v1          ; encoding: [0x00,0x00,0xec,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b64 a[6:7], v1

// GFX90A: ds_read_b64 a[6:7], v1 offset:4 ; encoding: [0x04,0x00,0xec,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b64 a[6:7], v1 offset:4

// GFX90A: ds_read2_b64 a[6:9], v1 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0xee,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b64 a[6:9], v1 offset0:127 offset1:255

// GFX90A: ds_read2_b64 a[252:255], v1 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0xee,0xda,0x01,0x00,0x00,0xfc]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b64 a[252:255], v1 offset0:127 offset1:255

// GFX90A: ds_read2_b64 a[6:9], v255 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0xee,0xda,0xff,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b64 a[6:9], v255 offset0:127 offset1:255

// GFX90A: ds_read2_b64 a[6:9], v1 offset1:255 ; encoding: [0x00,0xff,0xee,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b64 a[6:9], v1 offset1:255

// GFX90A: ds_read2_b64 a[6:9], v1 offset1:255 ; encoding: [0x00,0xff,0xee,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b64 a[6:9], v1 offset1:255

// GFX90A: ds_read2_b64 a[6:9], v1 offset0:16 offset1:255 ; encoding: [0x10,0xff,0xee,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b64 a[6:9], v1 offset0:16 offset1:255

// GFX90A: ds_read2_b64 a[6:9], v1 offset0:127 ; encoding: [0x7f,0x00,0xee,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b64 a[6:9], v1 offset0:127

// GFX90A: ds_read2_b64 a[6:9], v1 offset0:127 ; encoding: [0x7f,0x00,0xee,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b64 a[6:9], v1 offset0:127

// GFX90A: ds_read2_b64 a[6:9], v1 offset0:127 offset1:1 ; encoding: [0x7f,0x01,0xee,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2_b64 a[6:9], v1 offset0:127 offset1:1

// GFX90A: ds_read2st64_b64 a[6:9], v1 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0xf0,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b64 a[6:9], v1 offset0:127 offset1:255

// GFX90A: ds_read2st64_b64 a[252:255], v1 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0xf0,0xda,0x01,0x00,0x00,0xfc]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b64 a[252:255], v1 offset0:127 offset1:255

// GFX90A: ds_read2st64_b64 a[6:9], v255 offset0:127 offset1:255 ; encoding: [0x7f,0xff,0xf0,0xda,0xff,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b64 a[6:9], v255 offset0:127 offset1:255

// GFX90A: ds_read2st64_b64 a[6:9], v1 offset1:255 ; encoding: [0x00,0xff,0xf0,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b64 a[6:9], v1 offset1:255

// GFX90A: ds_read2st64_b64 a[6:9], v1 offset1:255 ; encoding: [0x00,0xff,0xf0,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b64 a[6:9], v1 offset1:255

// GFX90A: ds_read2st64_b64 a[6:9], v1 offset0:16 offset1:255 ; encoding: [0x10,0xff,0xf0,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b64 a[6:9], v1 offset0:16 offset1:255

// GFX90A: ds_read2st64_b64 a[6:9], v1 offset0:127 ; encoding: [0x7f,0x00,0xf0,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b64 a[6:9], v1 offset0:127

// GFX90A: ds_read2st64_b64 a[6:9], v1 offset0:127 ; encoding: [0x7f,0x00,0xf0,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b64 a[6:9], v1 offset0:127

// GFX90A: ds_read2st64_b64 a[6:9], v1 offset0:127 offset1:1 ; encoding: [0x7f,0x01,0xf0,0xda,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read2st64_b64 a[6:9], v1 offset0:127 offset1:1

// GFX90A: ds_condxchg32_rtn_b64 a[6:7], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xfc,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_condxchg32_rtn_b64 a[6:7], v1, a[2:3] offset:65535

// GFX90A: ds_condxchg32_rtn_b64 a[254:255], v1, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xfc,0xda,0x01,0x02,0x00,0xfe]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_condxchg32_rtn_b64 a[254:255], v1, a[2:3] offset:65535

// GFX90A: ds_condxchg32_rtn_b64 a[6:7], v255, a[2:3] offset:65535 ; encoding: [0xff,0xff,0xfc,0xda,0xff,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_condxchg32_rtn_b64 a[6:7], v255, a[2:3] offset:65535

// GFX90A: ds_condxchg32_rtn_b64 a[6:7], v1, a[254:255] offset:65535 ; encoding: [0xff,0xff,0xfc,0xda,0x01,0xfe,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_condxchg32_rtn_b64 a[6:7], v1, a[254:255] offset:65535

// GFX90A: ds_condxchg32_rtn_b64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xfc,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_condxchg32_rtn_b64 a[6:7], v1, a[2:3]

// GFX90A: ds_condxchg32_rtn_b64 a[6:7], v1, a[2:3] ; encoding: [0x00,0x00,0xfc,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_condxchg32_rtn_b64 a[6:7], v1, a[2:3]

// GFX90A: ds_condxchg32_rtn_b64 a[6:7], v1, a[2:3] offset:4 ; encoding: [0x04,0x00,0xfc,0xda,0x01,0x02,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
ds_condxchg32_rtn_b64 a[6:7], v1, a[2:3] offset:4

// GFX90A: ds_gws_init a0 offset:65535 gds ; encoding: [0xff,0xff,0x33,0xdb,0x00,0x00,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_gws_init a0 offset:65535 gds

// GFX90A: ds_gws_init a254 offset:65535 gds ; encoding: [0xff,0xff,0x33,0xdb,0xfe,0x00,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_gws_init a254 offset:65535 gds

// GFX90A: ds_gws_init a2 gds ; encoding: [0x00,0x00,0x33,0xdb,0x02,0x00,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_gws_init a2 gds

// GFX90A: ds_gws_init a0 gds ; encoding: [0x00,0x00,0x33,0xdb,0x00,0x00,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_gws_init a0 gds

// GFX90A: ds_gws_init a0 offset:4 gds ; encoding: [0x04,0x00,0x33,0xdb,0x00,0x00,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_gws_init a0 offset:4 gds

// GFX90A: ds_gws_sema_br a2 offset:65535 gds ; encoding: [0xff,0xff,0x37,0xdb,0x02,0x00,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_gws_sema_br a2 offset:65535 gds

// GFX90A: ds_gws_sema_br a254 offset:65535 gds ; encoding: [0xff,0xff,0x37,0xdb,0xfe,0x00,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_gws_sema_br a254 offset:65535 gds

// GFX90A: ds_gws_sema_br a0 gds ; encoding: [0x00,0x00,0x37,0xdb,0x00,0x00,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_gws_sema_br a0 gds

// GFX90A: ds_gws_sema_br a2 gds ; encoding: [0x00,0x00,0x37,0xdb,0x02,0x00,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_gws_sema_br a2 gds

// GFX90A: ds_gws_sema_br a0 offset:4 gds ; encoding: [0x04,0x00,0x37,0xdb,0x00,0x00,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_gws_sema_br a0 offset:4 gds

// GFX90A: ds_gws_barrier a2 offset:65535 gds ; encoding: [0xff,0xff,0x3b,0xdb,0x02,0x00,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_gws_barrier a2 offset:65535 gds

// GFX90A: ds_gws_barrier a254 offset:65535 gds ; encoding: [0xff,0xff,0x3b,0xdb,0xfe,0x00,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_gws_barrier a254 offset:65535 gds

// GFX90A: ds_gws_barrier a0 gds ; encoding: [0x00,0x00,0x3b,0xdb,0x00,0x00,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_gws_barrier a0 gds

// GFX90A: ds_gws_barrier a2 gds ; encoding: [0x00,0x00,0x3b,0xdb,0x02,0x00,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_gws_barrier a2 gds

// GFX90A: ds_gws_barrier a0 offset:4 gds ; encoding: [0x04,0x00,0x3b,0xdb,0x00,0x00,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_gws_barrier a0 offset:4 gds

// GFX90A: ds_consume a5 offset:65535      ; encoding: [0xff,0xff,0x7a,0xdb,0x00,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_consume a5 offset:65535

// GFX90A: ds_consume a255 offset:65535    ; encoding: [0xff,0xff,0x7a,0xdb,0x00,0x00,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_consume a255 offset:65535

// GFX90A: ds_consume a5                   ; encoding: [0x00,0x00,0x7a,0xdb,0x00,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_consume a5

// GFX90A: ds_consume a5                   ; encoding: [0x00,0x00,0x7a,0xdb,0x00,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_consume a5

// GFX90A: ds_consume a5 offset:4          ; encoding: [0x04,0x00,0x7a,0xdb,0x00,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_consume a5 offset:4

// GFX90A: ds_append a5 offset:65535       ; encoding: [0xff,0xff,0x7c,0xdb,0x00,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_append a5 offset:65535

// GFX90A: ds_append a255 offset:65535     ; encoding: [0xff,0xff,0x7c,0xdb,0x00,0x00,0x00,0xff]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_append a255 offset:65535

// GFX90A: ds_append a5                    ; encoding: [0x00,0x00,0x7c,0xdb,0x00,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_append a5

// GFX90A: ds_append a5                    ; encoding: [0x00,0x00,0x7c,0xdb,0x00,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_append a5

// GFX90A: ds_append a5 offset:4           ; encoding: [0x04,0x00,0x7c,0xdb,0x00,0x00,0x00,0x05]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_append a5 offset:4

// GFX90A: ds_write_b96 v1, a[2:4] offset:65535 ; encoding: [0xff,0xff,0xbc,0xdb,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b96 v1, a[2:4] offset:65535

// GFX90A: ds_write_b96 v255, a[2:4] offset:65535 ; encoding: [0xff,0xff,0xbc,0xdb,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b96 v255, a[2:4] offset:65535

// GFX90A: ds_write_b96 v1, a[252:254] offset:65535 ; encoding: [0xff,0xff,0xbc,0xdb,0x01,0xfc,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b96 v1, a[252:254] offset:65535

// GFX90A: ds_write_b96 v1, a[2:4]         ; encoding: [0x00,0x00,0xbc,0xdb,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b96 v1, a[2:4]

// GFX90A: ds_write_b96 v1, a[2:4]         ; encoding: [0x00,0x00,0xbc,0xdb,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b96 v1, a[2:4]

// GFX90A: ds_write_b96 v1, a[2:4] offset:4 ; encoding: [0x04,0x00,0xbc,0xdb,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b96 v1, a[2:4] offset:4

// GFX90A: ds_write_b128 v1, a[2:5] offset:65535 ; encoding: [0xff,0xff,0xbe,0xdb,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b128 v1, a[2:5] offset:65535

// GFX90A: ds_write_b128 v255, a[2:5] offset:65535 ; encoding: [0xff,0xff,0xbe,0xdb,0xff,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b128 v255, a[2:5] offset:65535

// GFX90A: ds_write_b128 v1, a[252:255] offset:65535 ; encoding: [0xff,0xff,0xbe,0xdb,0x01,0xfc,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b128 v1, a[252:255] offset:65535

// GFX90A: ds_write_b128 v1, a[2:5]        ; encoding: [0x00,0x00,0xbe,0xdb,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b128 v1, a[2:5]

// GFX90A: ds_write_b128 v1, a[2:5]        ; encoding: [0x00,0x00,0xbe,0xdb,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b128 v1, a[2:5]

// GFX90A: ds_write_b128 v1, a[2:5] offset:4 ; encoding: [0x04,0x00,0xbe,0xdb,0x01,0x02,0x00,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_write_b128 v1, a[2:5] offset:4

// GFX90A: ds_read_b96 a[6:8], v1 offset:65535 ; encoding: [0xff,0xff,0xfc,0xdb,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b96 a[6:8], v1 offset:65535

// GFX90A: ds_read_b96 a[252:254], v1 offset:65535 ; encoding: [0xff,0xff,0xfc,0xdb,0x01,0x00,0x00,0xfc]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b96 a[252:254], v1 offset:65535

// GFX90A: ds_read_b96 a[6:8], v255 offset:65535 ; encoding: [0xff,0xff,0xfc,0xdb,0xff,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b96 a[6:8], v255 offset:65535

// GFX90A: ds_read_b96 a[6:8], v1          ; encoding: [0x00,0x00,0xfc,0xdb,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b96 a[6:8], v1

// GFX90A: ds_read_b96 a[6:8], v1          ; encoding: [0x00,0x00,0xfc,0xdb,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b96 a[6:8], v1

// GFX90A: ds_read_b96 a[6:8], v1 offset:4 ; encoding: [0x04,0x00,0xfc,0xdb,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b96 a[6:8], v1 offset:4

// GFX90A: ds_read_b128 a[6:9], v1 offset:65535 ; encoding: [0xff,0xff,0xfe,0xdb,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b128 a[6:9], v1 offset:65535

// GFX90A: ds_read_b128 a[252:255], v1 offset:65535 ; encoding: [0xff,0xff,0xfe,0xdb,0x01,0x00,0x00,0xfc]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b128 a[252:255], v1 offset:65535

// GFX90A: ds_read_b128 a[6:9], v255 offset:65535 ; encoding: [0xff,0xff,0xfe,0xdb,0xff,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b128 a[6:9], v255 offset:65535

// GFX90A: ds_read_b128 a[6:9], v1         ; encoding: [0x00,0x00,0xfe,0xdb,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b128 a[6:9], v1

// GFX90A: ds_read_b128 a[6:9], v1         ; encoding: [0x00,0x00,0xfe,0xdb,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b128 a[6:9], v1

// GFX90A: ds_read_b128 a[6:9], v1 offset:4 ; encoding: [0x04,0x00,0xfe,0xdb,0x01,0x00,0x00,0x06]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: invalid register class: agpr loads and stores not supported on this GPU
ds_read_b128 a[6:9], v1 offset:4

// GFX90A: image_load a5, v[2:5], s[8:15] dmask:0x1 ; encoding: [0x00,0x01,0x01,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a5, v[2:5], s[8:15] dmask:0x1

// GFX90A: image_load a252, v[2:5], s[8:15] dmask:0x1 ; encoding: [0x00,0x01,0x01,0xf0,0x02,0xfc,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a252, v[2:5], s[8:15] dmask:0x1

// GFX90A: image_load a5, v[252:255], s[8:15] dmask:0x1 ; encoding: [0x00,0x01,0x01,0xf0,0xfc,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a5, v[252:255], s[8:15] dmask:0x1

// GFX90A: image_load a5, v[2:5], s[12:19] dmask:0x1 ; encoding: [0x00,0x01,0x01,0xf0,0x02,0x05,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a5, v[2:5], s[12:19] dmask:0x1

// GFX90A: image_load a5, v[2:5], s[92:99] dmask:0x1 ; encoding: [0x00,0x01,0x01,0xf0,0x02,0x05,0x17,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a5, v[2:5], s[92:99] dmask:0x1

// GFX90A: image_load a5, v[2:5], s[8:15] dmask:0x2 ; encoding: [0x00,0x02,0x01,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a5, v[2:5], s[8:15] dmask:0x2

// GFX90A: image_load a[6:7], v[2:5], s[8:15] dmask:0x3 ; encoding: [0x00,0x03,0x01,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a[6:7], v[2:5], s[8:15] dmask:0x3

// GFX90A: image_load a5, v[2:5], s[8:15] dmask:0x4 ; encoding: [0x00,0x04,0x01,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a5, v[2:5], s[8:15] dmask:0x4

// GFX90A: image_load a[6:7], v[2:5], s[8:15] dmask:0x5 ; encoding: [0x00,0x05,0x01,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a[6:7], v[2:5], s[8:15] dmask:0x5

// GFX90A: image_load a[6:7], v[2:5], s[8:15] dmask:0x6 ; encoding: [0x00,0x06,0x01,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a[6:7], v[2:5], s[8:15] dmask:0x6

// GFX90A: image_load a[6:8], v[2:5], s[8:15] dmask:0x7 ; encoding: [0x00,0x07,0x01,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a[6:8], v[2:5], s[8:15] dmask:0x7

// GFX90A: image_load a5, v[2:5], s[8:15] dmask:0x8 ; encoding: [0x00,0x08,0x01,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a5, v[2:5], s[8:15] dmask:0x8

// GFX90A: image_load a[6:7], v[2:5], s[8:15] dmask:0x9 ; encoding: [0x00,0x09,0x01,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a[6:7], v[2:5], s[8:15] dmask:0x9

// GFX90A: image_load a[6:7], v[2:5], s[8:15] dmask:0xa ; encoding: [0x00,0x0a,0x01,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a[6:7], v[2:5], s[8:15] dmask:0xa

// GFX90A: image_load a[6:8], v[2:5], s[8:15] dmask:0xb ; encoding: [0x00,0x0b,0x01,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a[6:8], v[2:5], s[8:15] dmask:0xb

// GFX90A: image_load a[6:7], v[2:5], s[8:15] dmask:0xc ; encoding: [0x00,0x0c,0x01,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a[6:7], v[2:5], s[8:15] dmask:0xc

// GFX90A: image_load a[6:8], v[2:5], s[8:15] dmask:0xd ; encoding: [0x00,0x0d,0x01,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a[6:8], v[2:5], s[8:15] dmask:0xd

// GFX90A: image_load a[6:8], v[2:5], s[8:15] dmask:0xe ; encoding: [0x00,0x0e,0x01,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a[6:8], v[2:5], s[8:15] dmask:0xe

// GFX90A: image_load a5, v[2:5], s[8:15]  ; encoding: [0x00,0x00,0x01,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a5, v[2:5], s[8:15]

// GFX90A: image_load a5, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x01,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a5, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_load a5, v[2:5], s[8:15] dmask:0x1 glc ; encoding: [0x00,0x21,0x01,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a5, v[2:5], s[8:15] dmask:0x1 glc

// GFX90A: image_load a5, v[2:5], s[8:15] dmask:0x1 slc ; encoding: [0x00,0x01,0x01,0xf2,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a5, v[2:5], s[8:15] dmask:0x1 slc

// GFX90A: image_load a5, v[2:5], s[8:15] dmask:0x1 lwe ; encoding: [0x00,0x01,0x03,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a5, v[2:5], s[8:15] dmask:0x1 lwe

// GFX90A: image_load a5, v[2:5], s[8:15] dmask:0x1 da ; encoding: [0x00,0x41,0x01,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a5, v[2:5], s[8:15] dmask:0x1 da

// GFX90A: image_load a5, v[2:5], s[8:15] dmask:0x1 d16 ; encoding: [0x00,0x01,0x01,0xf0,0x02,0x05,0x02,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_load a5, v[2:5], s[8:15] dmask:0x1 d16

// GFX90A: image_store a1, v[2:5], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x21,0xf0,0x02,0x01,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a1, v[2:5], s[12:19] dmask:0x1 unorm

// GFX90A: image_store a252, v[2:5], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x21,0xf0,0x02,0xfc,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a252, v[2:5], s[12:19] dmask:0x1 unorm

// GFX90A: image_store a1, v[252:255], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x21,0xf0,0xfc,0x01,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a1, v[252:255], s[12:19] dmask:0x1 unorm

// GFX90A: image_store a1, v[2:5], s[16:23] dmask:0x1 unorm ; encoding: [0x00,0x11,0x21,0xf0,0x02,0x01,0x04,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a1, v[2:5], s[16:23] dmask:0x1 unorm

// GFX90A: image_store a1, v[2:5], s[92:99] dmask:0x1 unorm ; encoding: [0x00,0x11,0x21,0xf0,0x02,0x01,0x17,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a1, v[2:5], s[92:99] dmask:0x1 unorm

// GFX90A: image_store a1, v[2:5], s[12:19] dmask:0x2 unorm ; encoding: [0x00,0x12,0x21,0xf0,0x02,0x01,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a1, v[2:5], s[12:19] dmask:0x2 unorm

// GFX90A: image_store a[2:3], v[2:5], s[12:19] dmask:0x3 unorm ; encoding: [0x00,0x13,0x21,0xf0,0x02,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a[2:3], v[2:5], s[12:19] dmask:0x3 unorm

// GFX90A: image_store a1, v[2:5], s[12:19] dmask:0x4 unorm ; encoding: [0x00,0x14,0x21,0xf0,0x02,0x01,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a1, v[2:5], s[12:19] dmask:0x4 unorm

// GFX90A: image_store a[2:3], v[2:5], s[12:19] dmask:0x5 unorm ; encoding: [0x00,0x15,0x21,0xf0,0x02,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a[2:3], v[2:5], s[12:19] dmask:0x5 unorm

// GFX90A: image_store a[2:3], v[2:5], s[12:19] dmask:0x6 unorm ; encoding: [0x00,0x16,0x21,0xf0,0x02,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a[2:3], v[2:5], s[12:19] dmask:0x6 unorm

// GFX90A: image_store a[2:4], v[2:5], s[12:19] dmask:0x7 unorm ; encoding: [0x00,0x17,0x21,0xf0,0x02,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a[2:4], v[2:5], s[12:19] dmask:0x7 unorm

// GFX90A: image_store a1, v[2:5], s[12:19] dmask:0x8 unorm ; encoding: [0x00,0x18,0x21,0xf0,0x02,0x01,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a1, v[2:5], s[12:19] dmask:0x8 unorm

// GFX90A: image_store a[2:3], v[2:5], s[12:19] dmask:0x9 unorm ; encoding: [0x00,0x19,0x21,0xf0,0x02,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a[2:3], v[2:5], s[12:19] dmask:0x9 unorm

// GFX90A: image_store a[2:3], v[2:5], s[12:19] dmask:0xa unorm ; encoding: [0x00,0x1a,0x21,0xf0,0x02,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a[2:3], v[2:5], s[12:19] dmask:0xa unorm

// GFX90A: image_store a[2:4], v[2:5], s[12:19] dmask:0xb unorm ; encoding: [0x00,0x1b,0x21,0xf0,0x02,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a[2:4], v[2:5], s[12:19] dmask:0xb unorm

// GFX90A: image_store a[2:3], v[2:5], s[12:19] dmask:0xc unorm ; encoding: [0x00,0x1c,0x21,0xf0,0x02,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a[2:3], v[2:5], s[12:19] dmask:0xc unorm

// GFX90A: image_store a[2:4], v[2:5], s[12:19] dmask:0xd unorm ; encoding: [0x00,0x1d,0x21,0xf0,0x02,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a[2:4], v[2:5], s[12:19] dmask:0xd unorm

// GFX90A: image_store a[2:4], v[2:5], s[12:19] dmask:0xe unorm ; encoding: [0x00,0x1e,0x21,0xf0,0x02,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a[2:4], v[2:5], s[12:19] dmask:0xe unorm

// GFX90A: image_store a[2:5], v[2:5], s[12:19] dmask:0xf unorm ; encoding: [0x00,0x1f,0x21,0xf0,0x02,0x02,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a[2:5], v[2:5], s[12:19] dmask:0xf unorm

// GFX90A: image_store a1, v[2:5], s[12:19] unorm ; encoding: [0x00,0x10,0x21,0xf0,0x02,0x01,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a1, v[2:5], s[12:19] unorm

// GFX90A: image_store a1, v[2:5], s[12:19] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x21,0xf0,0x02,0x01,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a1, v[2:5], s[12:19] dmask:0x1 unorm glc

// GFX90A: image_store a1, v[2:5], s[12:19] dmask:0x1 unorm slc ; encoding: [0x00,0x11,0x21,0xf2,0x02,0x01,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a1, v[2:5], s[12:19] dmask:0x1 unorm slc

// GFX90A: image_store a1, v[2:5], s[12:19] dmask:0x1 unorm lwe ; encoding: [0x00,0x11,0x23,0xf0,0x02,0x01,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a1, v[2:5], s[12:19] dmask:0x1 unorm lwe

// GFX90A: image_store a1, v[2:5], s[12:19] dmask:0x1 unorm da ; encoding: [0x00,0x51,0x21,0xf0,0x02,0x01,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a1, v[2:5], s[12:19] dmask:0x1 unorm da

// GFX90A: image_store a1, v[2:5], s[12:19] dmask:0x1 unorm d16 ; encoding: [0x00,0x11,0x21,0xf0,0x02,0x01,0x03,0x80]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_store a1, v[2:5], s[12:19] dmask:0x1 unorm d16

// GFX90A: image_atomic_swap a5, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x41,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_swap a5, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_swap a252, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x41,0xf0,0x02,0xfc,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_swap a252, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_swap a5, v[252:255], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x41,0xf0,0xfc,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_swap a5, v[252:255], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_swap a5, v[2:5], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x41,0xf0,0x02,0x05,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_swap a5, v[2:5], s[12:19] dmask:0x1 unorm

// GFX90A: image_atomic_swap a5, v[2:5], s[92:99] dmask:0x1 unorm ; encoding: [0x00,0x11,0x41,0xf0,0x02,0x05,0x17,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_swap a5, v[2:5], s[92:99] dmask:0x1 unorm

// GFX90A: image_atomic_swap a[6:7], v[2:5], s[8:15] dmask:0x3 unorm ; encoding: [0x00,0x13,0x41,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_swap a[6:7], v[2:5], s[8:15] dmask:0x3 unorm

// GFX90A: image_atomic_swap a5, v[2:5], s[8:15] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x41,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_swap a5, v[2:5], s[8:15] dmask:0x1 unorm glc

// GFX90A: image_atomic_swap a5, v[2:5], s[8:15] dmask:0x1 unorm slc ; encoding: [0x00,0x11,0x41,0xf2,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_swap a5, v[2:5], s[8:15] dmask:0x1 unorm slc

// GFX90A: image_atomic_swap a5, v[2:5], s[8:15] dmask:0x1 unorm lwe ; encoding: [0x00,0x11,0x43,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_swap a5, v[2:5], s[8:15] dmask:0x1 unorm lwe

// GFX90A: image_atomic_swap a5, v[2:5], s[8:15] dmask:0x1 unorm da ; encoding: [0x00,0x51,0x41,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_swap a5, v[2:5], s[8:15] dmask:0x1 unorm da

// GFX90A: image_atomic_cmpswap a[6:7], v[2:5], s[8:15] dmask:0x3 unorm ; encoding: [0x00,0x13,0x45,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_cmpswap a[6:7], v[2:5], s[8:15] dmask:0x3 unorm

// GFX90A: image_atomic_cmpswap a[252:253], v[2:5], s[8:15] dmask:0x3 unorm ; encoding: [0x00,0x13,0x45,0xf0,0x02,0xfc,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_cmpswap a[252:253], v[2:5], s[8:15] dmask:0x3 unorm

// GFX90A: image_atomic_cmpswap a[6:7], v[252:255], s[8:15] dmask:0x3 unorm ; encoding: [0x00,0x13,0x45,0xf0,0xfc,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_cmpswap a[6:7], v[252:255], s[8:15] dmask:0x3 unorm

// GFX90A: image_atomic_cmpswap a[6:7], v[2:5], s[12:19] dmask:0x3 unorm ; encoding: [0x00,0x13,0x45,0xf0,0x02,0x06,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_cmpswap a[6:7], v[2:5], s[12:19] dmask:0x3 unorm

// GFX90A: image_atomic_cmpswap a[6:7], v[2:5], s[92:99] dmask:0x3 unorm ; encoding: [0x00,0x13,0x45,0xf0,0x02,0x06,0x17,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_cmpswap a[6:7], v[2:5], s[92:99] dmask:0x3 unorm

// GFX90A: image_atomic_cmpswap a[6:9], v[2:5], s[8:15] dmask:0xf unorm ; encoding: [0x00,0x1f,0x45,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_cmpswap a[6:9], v[2:5], s[8:15] dmask:0xf unorm

// GFX90A: image_atomic_cmpswap a[6:7], v[2:5], s[8:15] dmask:0x3 unorm glc ; encoding: [0x00,0x33,0x45,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_cmpswap a[6:7], v[2:5], s[8:15] dmask:0x3 unorm glc

// GFX90A: image_atomic_cmpswap a[6:7], v[2:5], s[8:15] dmask:0x3 unorm slc ; encoding: [0x00,0x13,0x45,0xf2,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_cmpswap a[6:7], v[2:5], s[8:15] dmask:0x3 unorm slc

// GFX90A: image_atomic_cmpswap a[6:7], v[2:5], s[8:15] dmask:0x3 unorm lwe ; encoding: [0x00,0x13,0x47,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_cmpswap a[6:7], v[2:5], s[8:15] dmask:0x3 unorm lwe

// GFX90A: image_atomic_cmpswap a[6:7], v[2:5], s[8:15] dmask:0x3 unorm da ; encoding: [0x00,0x53,0x45,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_cmpswap a[6:7], v[2:5], s[8:15] dmask:0x3 unorm da

// GFX90A: image_atomic_add a5, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x49,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_add a5, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_add a252, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x49,0xf0,0x02,0xfc,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_add a252, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_add a5, v[252:255], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x49,0xf0,0xfc,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_add a5, v[252:255], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_add a5, v[2:5], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x49,0xf0,0x02,0x05,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_add a5, v[2:5], s[12:19] dmask:0x1 unorm

// GFX90A: image_atomic_add a5, v[2:5], s[92:99] dmask:0x1 unorm ; encoding: [0x00,0x11,0x49,0xf0,0x02,0x05,0x17,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_add a5, v[2:5], s[92:99] dmask:0x1 unorm

// GFX90A: image_atomic_add a[6:7], v[2:5], s[8:15] dmask:0x3 unorm ; encoding: [0x00,0x13,0x49,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_add a[6:7], v[2:5], s[8:15] dmask:0x3 unorm

// GFX90A: image_atomic_add a5, v[2:5], s[8:15] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x49,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_add a5, v[2:5], s[8:15] dmask:0x1 unorm glc

// GFX90A: image_atomic_add a5, v[2:5], s[8:15] dmask:0x1 unorm slc ; encoding: [0x00,0x11,0x49,0xf2,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_add a5, v[2:5], s[8:15] dmask:0x1 unorm slc

// GFX90A: image_atomic_add a5, v[2:5], s[8:15] dmask:0x1 unorm lwe ; encoding: [0x00,0x11,0x4b,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_add a5, v[2:5], s[8:15] dmask:0x1 unorm lwe

// GFX90A: image_atomic_add a5, v[2:5], s[8:15] dmask:0x1 unorm da ; encoding: [0x00,0x51,0x49,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_add a5, v[2:5], s[8:15] dmask:0x1 unorm da

// GFX90A: image_atomic_sub a5, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x4d,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_sub a5, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_sub a252, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x4d,0xf0,0x02,0xfc,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_sub a252, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_sub a5, v[252:255], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x4d,0xf0,0xfc,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_sub a5, v[252:255], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_sub a5, v[2:5], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x4d,0xf0,0x02,0x05,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_sub a5, v[2:5], s[12:19] dmask:0x1 unorm

// GFX90A: image_atomic_sub a5, v[2:5], s[92:99] dmask:0x1 unorm ; encoding: [0x00,0x11,0x4d,0xf0,0x02,0x05,0x17,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_sub a5, v[2:5], s[92:99] dmask:0x1 unorm

// GFX90A: image_atomic_sub a[6:7], v[2:5], s[8:15] dmask:0x3 unorm ; encoding: [0x00,0x13,0x4d,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_sub a[6:7], v[2:5], s[8:15] dmask:0x3 unorm

// GFX90A: image_atomic_sub a5, v[2:5], s[8:15] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x4d,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_sub a5, v[2:5], s[8:15] dmask:0x1 unorm glc

// GFX90A: image_atomic_sub a5, v[2:5], s[8:15] dmask:0x1 unorm slc ; encoding: [0x00,0x11,0x4d,0xf2,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_sub a5, v[2:5], s[8:15] dmask:0x1 unorm slc

// GFX90A: image_atomic_sub a5, v[2:5], s[8:15] dmask:0x1 unorm lwe ; encoding: [0x00,0x11,0x4f,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_sub a5, v[2:5], s[8:15] dmask:0x1 unorm lwe

// GFX90A: image_atomic_sub a5, v[2:5], s[8:15] dmask:0x1 unorm da ; encoding: [0x00,0x51,0x4d,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_sub a5, v[2:5], s[8:15] dmask:0x1 unorm da

// GFX90A: image_atomic_smin a5, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x51,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smin a5, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_smin a252, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x51,0xf0,0x02,0xfc,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smin a252, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_smin a5, v[252:255], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x51,0xf0,0xfc,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smin a5, v[252:255], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_smin a5, v[2:5], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x51,0xf0,0x02,0x05,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smin a5, v[2:5], s[12:19] dmask:0x1 unorm

// GFX90A: image_atomic_smin a5, v[2:5], s[92:99] dmask:0x1 unorm ; encoding: [0x00,0x11,0x51,0xf0,0x02,0x05,0x17,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smin a5, v[2:5], s[92:99] dmask:0x1 unorm

// GFX90A: image_atomic_smin a[6:7], v[2:5], s[8:15] dmask:0x3 unorm ; encoding: [0x00,0x13,0x51,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smin a[6:7], v[2:5], s[8:15] dmask:0x3 unorm

// GFX90A: image_atomic_smin a5, v[2:5], s[8:15] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x51,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smin a5, v[2:5], s[8:15] dmask:0x1 unorm glc

// GFX90A: image_atomic_smin a5, v[2:5], s[8:15] dmask:0x1 unorm slc ; encoding: [0x00,0x11,0x51,0xf2,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smin a5, v[2:5], s[8:15] dmask:0x1 unorm slc

// GFX90A: image_atomic_smin a5, v[2:5], s[8:15] dmask:0x1 unorm lwe ; encoding: [0x00,0x11,0x53,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smin a5, v[2:5], s[8:15] dmask:0x1 unorm lwe

// GFX90A: image_atomic_smin a5, v[2:5], s[8:15] dmask:0x1 unorm da ; encoding: [0x00,0x51,0x51,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smin a5, v[2:5], s[8:15] dmask:0x1 unorm da

// GFX90A: image_atomic_umin a5, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x55,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umin a5, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_umin a252, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x55,0xf0,0x02,0xfc,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umin a252, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_umin a5, v[252:255], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x55,0xf0,0xfc,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umin a5, v[252:255], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_umin a5, v[2:5], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x55,0xf0,0x02,0x05,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umin a5, v[2:5], s[12:19] dmask:0x1 unorm

// GFX90A: image_atomic_umin a5, v[2:5], s[92:99] dmask:0x1 unorm ; encoding: [0x00,0x11,0x55,0xf0,0x02,0x05,0x17,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umin a5, v[2:5], s[92:99] dmask:0x1 unorm

// GFX90A: image_atomic_umin a[6:7], v[2:5], s[8:15] dmask:0x3 unorm ; encoding: [0x00,0x13,0x55,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umin a[6:7], v[2:5], s[8:15] dmask:0x3 unorm

// GFX90A: image_atomic_umin a5, v[2:5], s[8:15] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x55,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umin a5, v[2:5], s[8:15] dmask:0x1 unorm glc

// GFX90A: image_atomic_umin a5, v[2:5], s[8:15] dmask:0x1 unorm slc ; encoding: [0x00,0x11,0x55,0xf2,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umin a5, v[2:5], s[8:15] dmask:0x1 unorm slc

// GFX90A: image_atomic_umin a5, v[2:5], s[8:15] dmask:0x1 unorm lwe ; encoding: [0x00,0x11,0x57,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umin a5, v[2:5], s[8:15] dmask:0x1 unorm lwe

// GFX90A: image_atomic_umin a5, v[2:5], s[8:15] dmask:0x1 unorm da ; encoding: [0x00,0x51,0x55,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umin a5, v[2:5], s[8:15] dmask:0x1 unorm da

// GFX90A: image_atomic_smax a5, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x59,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smax a5, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_smax a252, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x59,0xf0,0x02,0xfc,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smax a252, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_smax a5, v[252:255], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x59,0xf0,0xfc,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smax a5, v[252:255], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_smax a5, v[2:5], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x59,0xf0,0x02,0x05,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smax a5, v[2:5], s[12:19] dmask:0x1 unorm

// GFX90A: image_atomic_smax a5, v[2:5], s[92:99] dmask:0x1 unorm ; encoding: [0x00,0x11,0x59,0xf0,0x02,0x05,0x17,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smax a5, v[2:5], s[92:99] dmask:0x1 unorm

// GFX90A: image_atomic_smax a[6:7], v[2:5], s[8:15] dmask:0x3 unorm ; encoding: [0x00,0x13,0x59,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smax a[6:7], v[2:5], s[8:15] dmask:0x3 unorm

// GFX90A: image_atomic_smax a5, v[2:5], s[8:15] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x59,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smax a5, v[2:5], s[8:15] dmask:0x1 unorm glc

// GFX90A: image_atomic_smax a5, v[2:5], s[8:15] dmask:0x1 unorm slc ; encoding: [0x00,0x11,0x59,0xf2,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smax a5, v[2:5], s[8:15] dmask:0x1 unorm slc

// GFX90A: image_atomic_smax a5, v[2:5], s[8:15] dmask:0x1 unorm lwe ; encoding: [0x00,0x11,0x5b,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smax a5, v[2:5], s[8:15] dmask:0x1 unorm lwe

// GFX90A: image_atomic_smax a5, v[2:5], s[8:15] dmask:0x1 unorm da ; encoding: [0x00,0x51,0x59,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_smax a5, v[2:5], s[8:15] dmask:0x1 unorm da

// GFX90A: image_atomic_umax a5, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x5d,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umax a5, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_umax a252, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x5d,0xf0,0x02,0xfc,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umax a252, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_umax a5, v[252:255], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x5d,0xf0,0xfc,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umax a5, v[252:255], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_umax a5, v[2:5], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x5d,0xf0,0x02,0x05,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umax a5, v[2:5], s[12:19] dmask:0x1 unorm

// GFX90A: image_atomic_umax a5, v[2:5], s[92:99] dmask:0x1 unorm ; encoding: [0x00,0x11,0x5d,0xf0,0x02,0x05,0x17,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umax a5, v[2:5], s[92:99] dmask:0x1 unorm

// GFX90A: image_atomic_umax a[6:7], v[2:5], s[8:15] dmask:0x3 unorm ; encoding: [0x00,0x13,0x5d,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umax a[6:7], v[2:5], s[8:15] dmask:0x3 unorm

// GFX90A: image_atomic_umax a5, v[2:5], s[8:15] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x5d,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umax a5, v[2:5], s[8:15] dmask:0x1 unorm glc

// GFX90A: image_atomic_umax a5, v[2:5], s[8:15] dmask:0x1 unorm slc ; encoding: [0x00,0x11,0x5d,0xf2,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umax a5, v[2:5], s[8:15] dmask:0x1 unorm slc

// GFX90A: image_atomic_umax a5, v[2:5], s[8:15] dmask:0x1 unorm lwe ; encoding: [0x00,0x11,0x5f,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umax a5, v[2:5], s[8:15] dmask:0x1 unorm lwe

// GFX90A: image_atomic_umax a5, v[2:5], s[8:15] dmask:0x1 unorm da ; encoding: [0x00,0x51,0x5d,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_umax a5, v[2:5], s[8:15] dmask:0x1 unorm da

// GFX90A: image_atomic_and a5, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x61,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_and a5, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_and a252, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x61,0xf0,0x02,0xfc,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_and a252, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_and a5, v[252:255], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x61,0xf0,0xfc,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_and a5, v[252:255], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_and a5, v[2:5], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x61,0xf0,0x02,0x05,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_and a5, v[2:5], s[12:19] dmask:0x1 unorm

// GFX90A: image_atomic_and a5, v[2:5], s[92:99] dmask:0x1 unorm ; encoding: [0x00,0x11,0x61,0xf0,0x02,0x05,0x17,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_and a5, v[2:5], s[92:99] dmask:0x1 unorm

// GFX90A: image_atomic_and a[6:7], v[2:5], s[8:15] dmask:0x3 unorm ; encoding: [0x00,0x13,0x61,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_and a[6:7], v[2:5], s[8:15] dmask:0x3 unorm

// GFX90A: image_atomic_and a5, v[2:5], s[8:15] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x61,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_and a5, v[2:5], s[8:15] dmask:0x1 unorm glc

// GFX90A: image_atomic_and a5, v[2:5], s[8:15] dmask:0x1 unorm slc ; encoding: [0x00,0x11,0x61,0xf2,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_and a5, v[2:5], s[8:15] dmask:0x1 unorm slc

// GFX90A: image_atomic_and a5, v[2:5], s[8:15] dmask:0x1 unorm lwe ; encoding: [0x00,0x11,0x63,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_and a5, v[2:5], s[8:15] dmask:0x1 unorm lwe

// GFX90A: image_atomic_and a5, v[2:5], s[8:15] dmask:0x1 unorm da ; encoding: [0x00,0x51,0x61,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_and a5, v[2:5], s[8:15] dmask:0x1 unorm da

// GFX90A: image_atomic_or a5, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x65,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_or a5, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_or a252, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x65,0xf0,0x02,0xfc,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_or a252, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_or a5, v[252:255], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x65,0xf0,0xfc,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_or a5, v[252:255], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_or a5, v[2:5], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x65,0xf0,0x02,0x05,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_or a5, v[2:5], s[12:19] dmask:0x1 unorm

// GFX90A: image_atomic_or a5, v[2:5], s[92:99] dmask:0x1 unorm ; encoding: [0x00,0x11,0x65,0xf0,0x02,0x05,0x17,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_or a5, v[2:5], s[92:99] dmask:0x1 unorm

// GFX90A: image_atomic_or a[6:7], v[2:5], s[8:15] dmask:0x3 unorm ; encoding: [0x00,0x13,0x65,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_or a[6:7], v[2:5], s[8:15] dmask:0x3 unorm

// GFX90A: image_atomic_or a5, v[2:5], s[8:15] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x65,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_or a5, v[2:5], s[8:15] dmask:0x1 unorm glc

// GFX90A: image_atomic_or a5, v[2:5], s[8:15] dmask:0x1 unorm slc ; encoding: [0x00,0x11,0x65,0xf2,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_or a5, v[2:5], s[8:15] dmask:0x1 unorm slc

// GFX90A: image_atomic_or a5, v[2:5], s[8:15] dmask:0x1 unorm lwe ; encoding: [0x00,0x11,0x67,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_or a5, v[2:5], s[8:15] dmask:0x1 unorm lwe

// GFX90A: image_atomic_or a5, v[2:5], s[8:15] dmask:0x1 unorm da ; encoding: [0x00,0x51,0x65,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_or a5, v[2:5], s[8:15] dmask:0x1 unorm da

// GFX90A: image_atomic_xor a5, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x69,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_xor a5, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_xor a252, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x69,0xf0,0x02,0xfc,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_xor a252, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_xor a5, v[252:255], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x69,0xf0,0xfc,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_xor a5, v[252:255], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_xor a5, v[2:5], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x69,0xf0,0x02,0x05,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_xor a5, v[2:5], s[12:19] dmask:0x1 unorm

// GFX90A: image_atomic_xor a5, v[2:5], s[92:99] dmask:0x1 unorm ; encoding: [0x00,0x11,0x69,0xf0,0x02,0x05,0x17,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_xor a5, v[2:5], s[92:99] dmask:0x1 unorm

// GFX90A: image_atomic_xor a[6:7], v[2:5], s[8:15] dmask:0x3 unorm ; encoding: [0x00,0x13,0x69,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_xor a[6:7], v[2:5], s[8:15] dmask:0x3 unorm

// GFX90A: image_atomic_xor a5, v[2:5], s[8:15] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x69,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_xor a5, v[2:5], s[8:15] dmask:0x1 unorm glc

// GFX90A: image_atomic_xor a5, v[2:5], s[8:15] dmask:0x1 unorm slc ; encoding: [0x00,0x11,0x69,0xf2,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_xor a5, v[2:5], s[8:15] dmask:0x1 unorm slc

// GFX90A: image_atomic_xor a5, v[2:5], s[8:15] dmask:0x1 unorm lwe ; encoding: [0x00,0x11,0x6b,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_xor a5, v[2:5], s[8:15] dmask:0x1 unorm lwe

// GFX90A: image_atomic_xor a5, v[2:5], s[8:15] dmask:0x1 unorm da ; encoding: [0x00,0x51,0x69,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_xor a5, v[2:5], s[8:15] dmask:0x1 unorm da

// GFX90A: image_atomic_inc a5, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x6d,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_inc a5, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_inc a252, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x6d,0xf0,0x02,0xfc,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_inc a252, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_inc a5, v[252:255], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x6d,0xf0,0xfc,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_inc a5, v[252:255], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_inc a5, v[2:5], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x6d,0xf0,0x02,0x05,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_inc a5, v[2:5], s[12:19] dmask:0x1 unorm

// GFX90A: image_atomic_inc a5, v[2:5], s[92:99] dmask:0x1 unorm ; encoding: [0x00,0x11,0x6d,0xf0,0x02,0x05,0x17,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_inc a5, v[2:5], s[92:99] dmask:0x1 unorm

// GFX90A: image_atomic_inc a[6:7], v[2:5], s[8:15] dmask:0x3 unorm ; encoding: [0x00,0x13,0x6d,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_inc a[6:7], v[2:5], s[8:15] dmask:0x3 unorm

// GFX90A: image_atomic_inc a5, v[2:5], s[8:15] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x6d,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_inc a5, v[2:5], s[8:15] dmask:0x1 unorm glc

// GFX90A: image_atomic_inc a5, v[2:5], s[8:15] dmask:0x1 unorm slc ; encoding: [0x00,0x11,0x6d,0xf2,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_inc a5, v[2:5], s[8:15] dmask:0x1 unorm slc

// GFX90A: image_atomic_inc a5, v[2:5], s[8:15] dmask:0x1 unorm lwe ; encoding: [0x00,0x11,0x6f,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_inc a5, v[2:5], s[8:15] dmask:0x1 unorm lwe

// GFX90A: image_atomic_inc a5, v[2:5], s[8:15] dmask:0x1 unorm da ; encoding: [0x00,0x51,0x6d,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_inc a5, v[2:5], s[8:15] dmask:0x1 unorm da

// GFX90A: image_atomic_dec a5, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x71,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_dec a5, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_dec a252, v[2:5], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x71,0xf0,0x02,0xfc,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_dec a252, v[2:5], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_dec a5, v[252:255], s[8:15] dmask:0x1 unorm ; encoding: [0x00,0x11,0x71,0xf0,0xfc,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_dec a5, v[252:255], s[8:15] dmask:0x1 unorm

// GFX90A: image_atomic_dec a5, v[2:5], s[12:19] dmask:0x1 unorm ; encoding: [0x00,0x11,0x71,0xf0,0x02,0x05,0x03,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_dec a5, v[2:5], s[12:19] dmask:0x1 unorm

// GFX90A: image_atomic_dec a5, v[2:5], s[92:99] dmask:0x1 unorm ; encoding: [0x00,0x11,0x71,0xf0,0x02,0x05,0x17,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_dec a5, v[2:5], s[92:99] dmask:0x1 unorm

// GFX90A: image_atomic_dec a[6:7], v[2:5], s[8:15] dmask:0x3 unorm ; encoding: [0x00,0x13,0x71,0xf0,0x02,0x06,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_dec a[6:7], v[2:5], s[8:15] dmask:0x3 unorm

// GFX90A: image_atomic_dec a5, v[2:5], s[8:15] dmask:0x1 unorm glc ; encoding: [0x00,0x31,0x71,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_dec a5, v[2:5], s[8:15] dmask:0x1 unorm glc

// GFX90A: image_atomic_dec a5, v[2:5], s[8:15] dmask:0x1 unorm slc ; encoding: [0x00,0x11,0x71,0xf2,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_dec a5, v[2:5], s[8:15] dmask:0x1 unorm slc

// GFX90A: image_atomic_dec a5, v[2:5], s[8:15] dmask:0x1 unorm lwe ; encoding: [0x00,0x11,0x73,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_dec a5, v[2:5], s[8:15] dmask:0x1 unorm lwe

// GFX90A: image_atomic_dec a5, v[2:5], s[8:15] dmask:0x1 unorm da ; encoding: [0x00,0x51,0x71,0xf0,0x02,0x05,0x02,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_atomic_dec a5, v[2:5], s[8:15] dmask:0x1 unorm da

// GFX90A: image_sample a5, v[0:3], s[8:15], s[12:15] dmask:0x1 ; encoding: [0x00,0x01,0x81,0xf0,0x00,0x05,0x62,0x00]
// NOT-GFX90A: :[[@LINE+1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
image_sample a5, v[0:3], s[8:15], s[12:15] dmask:0x1
