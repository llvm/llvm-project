// RUN: llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s | FileCheck --check-prefixes=GFX1210 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 %s 2>&1 | FileCheck --check-prefix=GFX1200-ERR --implicit-check-not=error: %s

v_fmac_f64 v[4:5], v[2:3], v[4:5]
// GFX1210: v_fmac_f64_e32 v[4:5], v[2:3], v[4:5]   ; encoding: [0x02,0x09,0x08,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[254:255], v[2:3], v[4:5]
// GFX1210: v_fmac_f64_e32 v[254:255], v[2:3], v[4:5] ; encoding: [0x02,0x09,0xfc,0x2f]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[254:255], v[4:5]
// GFX1210: v_fmac_f64_e32 v[4:5], v[254:255], v[4:5] ; encoding: [0xfe,0x09,0x08,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], vcc, v[4:5]
// GFX1210: v_fmac_f64_e32 v[4:5], vcc, v[4:5]      ; encoding: [0x6a,0x08,0x08,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], exec, v[4:5]
// GFX1210: v_fmac_f64_e32 v[4:5], exec, v[4:5]     ; encoding: [0x7e,0x08,0x08,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], 0, v[4:5]
// GFX1210: v_fmac_f64_e32 v[4:5], 0, v[4:5]        ; encoding: [0x80,0x08,0x08,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], -1, v[4:5]
// GFX1210: v_fmac_f64_e32 v[4:5], -1, v[4:5]       ; encoding: [0xc1,0x08,0x08,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], 0.5, v[4:5]
// GFX1210: v_fmac_f64_e32 v[4:5], 0.5, v[4:5]      ; encoding: [0xf0,0x08,0x08,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], -4.0, v[4:5]
// GFX1210: v_fmac_f64_e32 v[4:5], -4.0, v[4:5]     ; encoding: [0xf7,0x08,0x08,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], 0xaf123456, v[4:5]
// GFX1210: v_fmac_f64_e32 v[4:5], 0xaf123456, v[4:5] ; encoding: [0xff,0x08,0x08,0x2e,0x56,0x34,0x12,0xaf]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], 0x3f717273, v[4:5]
// GFX1210: v_fmac_f64_e32 v[4:5], 0x3f717273, v[4:5] ; encoding: [0xff,0x08,0x08,0x2e,0x73,0x72,0x71,0x3f]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[2:3], v[254:255]
// GFX1210: v_fmac_f64_e32 v[4:5], v[2:3], v[254:255] ; encoding: [0x02,0xfd,0x09,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[2:3], v[8:9]
// GFX1210: v_fmac_f64_e32 v[4:5], v[2:3], v[8:9]   ; encoding: [0x02,0x11,0x08,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[254:255], v[2:3], v[8:9]
// GFX1210: v_fmac_f64_e32 v[254:255], v[2:3], v[8:9] ; encoding: [0x02,0x11,0xfc,0x2f]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[254:255], v[8:9]
// GFX1210: v_fmac_f64_e32 v[4:5], v[254:255], v[8:9] ; encoding: [0xfe,0x11,0x08,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], vcc, v[8:9]
// GFX1210: v_fmac_f64_e32 v[4:5], vcc, v[8:9]      ; encoding: [0x6a,0x10,0x08,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], exec, v[8:9]
// GFX1210: v_fmac_f64_e32 v[4:5], exec, v[8:9]     ; encoding: [0x7e,0x10,0x08,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], 0, v[8:9]
// GFX1210: v_fmac_f64_e32 v[4:5], 0, v[8:9]        ; encoding: [0x80,0x10,0x08,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], -1, v[8:9]
// GFX1210: v_fmac_f64_e32 v[4:5], -1, v[8:9]       ; encoding: [0xc1,0x10,0x08,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], 0.5, v[8:9]
// GFX1210: v_fmac_f64_e32 v[4:5], 0.5, v[8:9]      ; encoding: [0xf0,0x10,0x08,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], -4.0, v[8:9]
// GFX1210: v_fmac_f64_e32 v[4:5], -4.0, v[8:9]     ; encoding: [0xf7,0x10,0x08,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[2:3], v[254:255]
// GFX1210: v_fmac_f64_e32 v[4:5], v[2:3], v[254:255] ; encoding: [0x02,0xfd,0x09,0x2e]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[2:3], vcc
// GFX1210: v_fmac_f64_e64 v[4:5], v[2:3], vcc      ; encoding: [0x04,0x00,0x17,0xd5,0x02,0xd5,0x00,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[2:3], exec
// GFX1210: v_fmac_f64_e64 v[4:5], v[2:3], exec     ; encoding: [0x04,0x00,0x17,0xd5,0x02,0xfd,0x00,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[2:3], 0
// GFX1210: v_fmac_f64_e64 v[4:5], v[2:3], 0        ; encoding: [0x04,0x00,0x17,0xd5,0x02,0x01,0x01,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[2:3], -1
// GFX1210: v_fmac_f64_e64 v[4:5], v[2:3], -1       ; encoding: [0x04,0x00,0x17,0xd5,0x02,0x83,0x01,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[2:3], 0.5
// GFX1210: v_fmac_f64_e64 v[4:5], v[2:3], 0.5      ; encoding: [0x04,0x00,0x17,0xd5,0x02,0xe1,0x01,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[2:3], -4.0
// GFX1210: v_fmac_f64_e64 v[4:5], v[2:3], -4.0     ; encoding: [0x04,0x00,0x17,0xd5,0x02,0xef,0x01,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], -v[2:3], v[8:9]
// GFX1210: v_fmac_f64_e64 v[4:5], -v[2:3], v[8:9]  ; encoding: [0x04,0x00,0x17,0xd5,0x02,0x11,0x02,0x20]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[2:3], -v[8:9]
// GFX1210: v_fmac_f64_e64 v[4:5], v[2:3], -v[8:9]  ; encoding: [0x04,0x00,0x17,0xd5,0x02,0x11,0x02,0x40]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], -v[2:3], -v[8:9]
// GFX1210: v_fmac_f64_e64 v[4:5], -v[2:3], -v[8:9] ; encoding: [0x04,0x00,0x17,0xd5,0x02,0x11,0x02,0x60]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], |v[2:3]|, v[8:9]
// GFX1210: v_fmac_f64_e64 v[4:5], |v[2:3]|, v[8:9] ; encoding: [0x04,0x01,0x17,0xd5,0x02,0x11,0x02,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[2:3], |v[8:9]|
// GFX1210: v_fmac_f64_e64 v[4:5], v[2:3], |v[8:9]| ; encoding: [0x04,0x02,0x17,0xd5,0x02,0x11,0x02,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], |v[2:3]|, |v[8:9]|
// GFX1210: v_fmac_f64_e64 v[4:5], |v[2:3]|, |v[8:9]| ; encoding: [0x04,0x03,0x17,0xd5,0x02,0x11,0x02,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[2:3], v[8:9] clamp
// GFX1210: v_fmac_f64_e64 v[4:5], v[2:3], v[8:9] clamp ; encoding: [0x04,0x80,0x17,0xd5,0x02,0x11,0x02,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[2:3], v[8:9] mul:2
// GFX1210: v_fmac_f64_e64 v[4:5], v[2:3], v[8:9] mul:2 ; encoding: [0x04,0x00,0x17,0xd5,0x02,0x11,0x02,0x08]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[2:3], v[8:9] mul:4
// GFX1210: v_fmac_f64_e64 v[4:5], v[2:3], v[8:9] mul:4 ; encoding: [0x04,0x00,0x17,0xd5,0x02,0x11,0x02,0x10]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_fmac_f64 v[4:5], v[2:3], v[8:9] div:2
// GFX1210: v_fmac_f64_e64 v[4:5], v[2:3], v[8:9] div:2 ; encoding: [0x04,0x00,0x17,0xd5,0x02,0x11,0x02,0x18]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], v[2:3], v[4:5]
// GFX1210: v_add_nc_u64_e32 v[4:5], v[2:3], v[4:5] ; encoding: [0x02,0x09,0x08,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[254:255], v[2:3], v[4:5]
// GFX1210: v_add_nc_u64_e32 v[254:255], v[2:3], v[4:5] ; encoding: [0x02,0x09,0xfc,0x51]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64_e64 v[4:5], s[2:3], s[4:5]
// GFX1210: v_add_nc_u64_e64 v[4:5], s[2:3], s[4:5] ; encoding: [0x04,0x00,0x28,0xd5,0x02,0x08,0x00,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], v[254:255], v[4:5]
// GFX1210: v_add_nc_u64_e32 v[4:5], v[254:255], v[4:5] ; encoding: [0xfe,0x09,0x08,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], vcc, v[4:5]
// GFX1210: v_add_nc_u64_e32 v[4:5], vcc, v[4:5]    ; encoding: [0x6a,0x08,0x08,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], exec, v[4:5]
// GFX1210: v_add_nc_u64_e32 v[4:5], exec, v[4:5]   ; encoding: [0x7e,0x08,0x08,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], 0, v[4:5]
// GFX1210: v_add_nc_u64_e32 v[4:5], 0, v[4:5]      ; encoding: [0x80,0x08,0x08,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], -1, v[4:5]
// GFX1210: v_add_nc_u64_e32 v[4:5], -1, v[4:5]     ; encoding: [0xc1,0x08,0x08,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], 0.5, v[4:5]
// GFX1210: v_add_nc_u64_e32 v[4:5], 0.5, v[4:5]    ; encoding: [0xf0,0x08,0x08,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], -4.0, v[4:5]
// GFX1210: v_add_nc_u64_e32 v[4:5], -4.0, v[4:5]   ; encoding: [0xf7,0x08,0x08,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], 0xaf123456, v[4:5]
// GFX1210: v_add_nc_u64_e32 v[4:5], 0xaf123456, v[4:5] ; encoding: [0xff,0x08,0x08,0x50,0x56,0x34,0x12,0xaf]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], 0x3f717273, v[4:5]
// GFX1210: v_add_nc_u64_e32 v[4:5], 0x3f717273, v[4:5] ; encoding: [0xff,0x08,0x08,0x50,0x73,0x72,0x71,0x3f]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], v[2:3], v[254:255]
// GFX1210: v_add_nc_u64_e32 v[4:5], v[2:3], v[254:255] ; encoding: [0x02,0xfd,0x09,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], v[2:3], v[8:9]
// GFX1210: v_add_nc_u64_e32 v[4:5], v[2:3], v[8:9] ; encoding: [0x02,0x11,0x08,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[254:255], v[2:3], v[8:9]
// GFX1210: v_add_nc_u64_e32 v[254:255], v[2:3], v[8:9] ; encoding: [0x02,0x11,0xfc,0x51]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], v[254:255], v[8:9]
// GFX1210: v_add_nc_u64_e32 v[4:5], v[254:255], v[8:9] ; encoding: [0xfe,0x11,0x08,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], vcc, v[8:9]
// GFX1210: v_add_nc_u64_e32 v[4:5], vcc, v[8:9]    ; encoding: [0x6a,0x10,0x08,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], exec, v[8:9]
// GFX1210: v_add_nc_u64_e32 v[4:5], exec, v[8:9]   ; encoding: [0x7e,0x10,0x08,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], 0, v[8:9]
// GFX1210: v_add_nc_u64_e32 v[4:5], 0, v[8:9]      ; encoding: [0x80,0x10,0x08,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], -1, v[8:9]
// GFX1210: v_add_nc_u64_e32 v[4:5], -1, v[8:9]     ; encoding: [0xc1,0x10,0x08,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], 0.5, v[8:9]
// GFX1210: v_add_nc_u64_e32 v[4:5], 0.5, v[8:9]    ; encoding: [0xf0,0x10,0x08,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], -4.0, v[8:9]
// GFX1210: v_add_nc_u64_e32 v[4:5], -4.0, v[8:9]   ; encoding: [0xf7,0x10,0x08,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], v[2:3], v[254:255]
// GFX1210: v_add_nc_u64_e32 v[4:5], v[2:3], v[254:255] ; encoding: [0x02,0xfd,0x09,0x50]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], v[2:3], vcc
// GFX1210: v_add_nc_u64_e64 v[4:5], v[2:3], vcc    ; encoding: [0x04,0x00,0x28,0xd5,0x02,0xd5,0x00,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], v[2:3], exec
// GFX1210: v_add_nc_u64_e64 v[4:5], v[2:3], exec   ; encoding: [0x04,0x00,0x28,0xd5,0x02,0xfd,0x00,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], v[2:3], 0
// GFX1210: v_add_nc_u64_e64 v[4:5], v[2:3], 0      ; encoding: [0x04,0x00,0x28,0xd5,0x02,0x01,0x01,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], v[2:3], -1
// GFX1210: v_add_nc_u64_e64 v[4:5], v[2:3], -1     ; encoding: [0x04,0x00,0x28,0xd5,0x02,0x83,0x01,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], v[2:3], 0.5
// GFX1210: v_add_nc_u64_e64 v[4:5], v[2:3], 0.5    ; encoding: [0x04,0x00,0x28,0xd5,0x02,0xe1,0x01,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], v[2:3], -4.0
// GFX1210: v_add_nc_u64_e64 v[4:5], v[2:3], -4.0   ; encoding: [0x04,0x00,0x28,0xd5,0x02,0xef,0x01,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_add_nc_u64 v[4:5], v[2:3], v[8:9] clamp
// GFX1210: v_add_nc_u64_e64 v[4:5], v[2:3], v[8:9] clamp ; encoding: [0x04,0x80,0x28,0xd5,0x02,0x11,0x02,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], v[2:3], v[4:5]
// GFX1210: v_sub_nc_u64_e32 v[4:5], v[2:3], v[4:5] ; encoding: [0x02,0x09,0x08,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[254:255], v[2:3], v[4:5]
// GFX1210: v_sub_nc_u64_e32 v[254:255], v[2:3], v[4:5] ; encoding: [0x02,0x09,0xfc,0x53]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64_e64 v[4:5], s[2:3], s[4:5]
// GFX1210: v_sub_nc_u64_e64 v[4:5], s[2:3], s[4:5] ; encoding: [0x04,0x00,0x29,0xd5,0x02,0x08,0x00,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], v[254:255], v[4:5]
// GFX1210: v_sub_nc_u64_e32 v[4:5], v[254:255], v[4:5] ; encoding: [0xfe,0x09,0x08,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], vcc, v[4:5]
// GFX1210: v_sub_nc_u64_e32 v[4:5], vcc, v[4:5]    ; encoding: [0x6a,0x08,0x08,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], exec, v[4:5]
// GFX1210: v_sub_nc_u64_e32 v[4:5], exec, v[4:5]   ; encoding: [0x7e,0x08,0x08,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], 0, v[4:5]
// GFX1210: v_sub_nc_u64_e32 v[4:5], 0, v[4:5]      ; encoding: [0x80,0x08,0x08,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], -1, v[4:5]
// GFX1210: v_sub_nc_u64_e32 v[4:5], -1, v[4:5]     ; encoding: [0xc1,0x08,0x08,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], 0.5, v[4:5]
// GFX1210: v_sub_nc_u64_e32 v[4:5], 0.5, v[4:5]    ; encoding: [0xf0,0x08,0x08,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], -4.0, v[4:5]
// GFX1210: v_sub_nc_u64_e32 v[4:5], -4.0, v[4:5]   ; encoding: [0xf7,0x08,0x08,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], 0xaf123456, v[4:5]
// GFX1210: v_sub_nc_u64_e32 v[4:5], 0xaf123456, v[4:5] ; encoding: [0xff,0x08,0x08,0x52,0x56,0x34,0x12,0xaf]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], 0x3f717273, v[4:5]
// GFX1210: v_sub_nc_u64_e32 v[4:5], 0x3f717273, v[4:5] ; encoding: [0xff,0x08,0x08,0x52,0x73,0x72,0x71,0x3f]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], v[2:3], v[254:255]
// GFX1210: v_sub_nc_u64_e32 v[4:5], v[2:3], v[254:255] ; encoding: [0x02,0xfd,0x09,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], v[2:3], v[8:9]
// GFX1210: v_sub_nc_u64_e32 v[4:5], v[2:3], v[8:9] ; encoding: [0x02,0x11,0x08,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[254:255], v[2:3], v[8:9]
// GFX1210: v_sub_nc_u64_e32 v[254:255], v[2:3], v[8:9] ; encoding: [0x02,0x11,0xfc,0x53]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], v[254:255], v[8:9]
// GFX1210: v_sub_nc_u64_e32 v[4:5], v[254:255], v[8:9] ; encoding: [0xfe,0x11,0x08,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], vcc, v[8:9]
// GFX1210: v_sub_nc_u64_e32 v[4:5], vcc, v[8:9]    ; encoding: [0x6a,0x10,0x08,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], exec, v[8:9]
// GFX1210: v_sub_nc_u64_e32 v[4:5], exec, v[8:9]   ; encoding: [0x7e,0x10,0x08,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], 0, v[8:9]
// GFX1210: v_sub_nc_u64_e32 v[4:5], 0, v[8:9]      ; encoding: [0x80,0x10,0x08,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], -1, v[8:9]
// GFX1210: v_sub_nc_u64_e32 v[4:5], -1, v[8:9]     ; encoding: [0xc1,0x10,0x08,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], 0.5, v[8:9]
// GFX1210: v_sub_nc_u64_e32 v[4:5], 0.5, v[8:9]    ; encoding: [0xf0,0x10,0x08,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], -4.0, v[8:9]
// GFX1210: v_sub_nc_u64_e32 v[4:5], -4.0, v[8:9]   ; encoding: [0xf7,0x10,0x08,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], v[2:3], v[254:255]
// GFX1210: v_sub_nc_u64_e32 v[4:5], v[2:3], v[254:255] ; encoding: [0x02,0xfd,0x09,0x52]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], v[2:3], vcc
// GFX1210: v_sub_nc_u64_e64 v[4:5], v[2:3], vcc    ; encoding: [0x04,0x00,0x29,0xd5,0x02,0xd5,0x00,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], v[2:3], exec
// GFX1210: v_sub_nc_u64_e64 v[4:5], v[2:3], exec   ; encoding: [0x04,0x00,0x29,0xd5,0x02,0xfd,0x00,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], v[2:3], 0
// GFX1210: v_sub_nc_u64_e64 v[4:5], v[2:3], 0      ; encoding: [0x04,0x00,0x29,0xd5,0x02,0x01,0x01,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], v[2:3], -1
// GFX1210: v_sub_nc_u64_e64 v[4:5], v[2:3], -1     ; encoding: [0x04,0x00,0x29,0xd5,0x02,0x83,0x01,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], v[2:3], 0.5
// GFX1210: v_sub_nc_u64_e64 v[4:5], v[2:3], 0.5    ; encoding: [0x04,0x00,0x29,0xd5,0x02,0xe1,0x01,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], v[2:3], -4.0
// GFX1210: v_sub_nc_u64_e64 v[4:5], v[2:3], -4.0   ; encoding: [0x04,0x00,0x29,0xd5,0x02,0xef,0x01,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_sub_nc_u64 v[4:5], v[2:3], v[8:9] clamp
// GFX1210: v_sub_nc_u64_e64 v[4:5], v[2:3], v[8:9] clamp ; encoding: [0x04,0x80,0x29,0xd5,0x02,0x11,0x02,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], v[2:3], v[4:5]
// GFX1210: v_mul_u64_e32 v[4:5], v[2:3], v[4:5]    ; encoding: [0x02,0x09,0x08,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[254:255], v[2:3], v[4:5]
// GFX1210: v_mul_u64_e32 v[254:255], v[2:3], v[4:5] ; encoding: [0x02,0x09,0xfc,0x55]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64_e64 v[4:5], s[2:3], s[4:5]
// GFX1210: v_mul_u64_e64 v[4:5], s[2:3], s[4:5]    ; encoding: [0x04,0x00,0x2a,0xd5,0x02,0x08,0x00,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], v[254:255], v[4:5]
// GFX1210: v_mul_u64_e32 v[4:5], v[254:255], v[4:5] ; encoding: [0xfe,0x09,0x08,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], vcc, v[4:5]
// GFX1210: v_mul_u64_e32 v[4:5], vcc, v[4:5]       ; encoding: [0x6a,0x08,0x08,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], exec, v[4:5]
// GFX1210: v_mul_u64_e32 v[4:5], exec, v[4:5]      ; encoding: [0x7e,0x08,0x08,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], 0, v[4:5]
// GFX1210: v_mul_u64_e32 v[4:5], 0, v[4:5]         ; encoding: [0x80,0x08,0x08,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], -1, v[4:5]
// GFX1210: v_mul_u64_e32 v[4:5], -1, v[4:5]        ; encoding: [0xc1,0x08,0x08,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], 0.5, v[4:5]
// GFX1210: v_mul_u64_e32 v[4:5], 0.5, v[4:5]       ; encoding: [0xf0,0x08,0x08,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], -4.0, v[4:5]
// GFX1210: v_mul_u64_e32 v[4:5], -4.0, v[4:5]      ; encoding: [0xf7,0x08,0x08,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], 0xaf123456, v[4:5]
// GFX1210: v_mul_u64_e32 v[4:5], 0xaf123456, v[4:5] ; encoding: [0xff,0x08,0x08,0x54,0x56,0x34,0x12,0xaf]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], 0x3f717273, v[4:5]
// GFX1210: v_mul_u64_e32 v[4:5], 0x3f717273, v[4:5] ; encoding: [0xff,0x08,0x08,0x54,0x73,0x72,0x71,0x3f]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], v[2:3], v[254:255]
// GFX1210: v_mul_u64_e32 v[4:5], v[2:3], v[254:255] ; encoding: [0x02,0xfd,0x09,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], v[2:3], v[8:9]
// GFX1210: v_mul_u64_e32 v[4:5], v[2:3], v[8:9]    ; encoding: [0x02,0x11,0x08,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[254:255], v[2:3], v[8:9]
// GFX1210: v_mul_u64_e32 v[254:255], v[2:3], v[8:9] ; encoding: [0x02,0x11,0xfc,0x55]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], v[254:255], v[8:9]
// GFX1210: v_mul_u64_e32 v[4:5], v[254:255], v[8:9] ; encoding: [0xfe,0x11,0x08,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], vcc, v[8:9]
// GFX1210: v_mul_u64_e32 v[4:5], vcc, v[8:9]       ; encoding: [0x6a,0x10,0x08,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], exec, v[8:9]
// GFX1210: v_mul_u64_e32 v[4:5], exec, v[8:9]      ; encoding: [0x7e,0x10,0x08,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], 0, v[8:9]
// GFX1210: v_mul_u64_e32 v[4:5], 0, v[8:9]         ; encoding: [0x80,0x10,0x08,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], -1, v[8:9]
// GFX1210: v_mul_u64_e32 v[4:5], -1, v[8:9]        ; encoding: [0xc1,0x10,0x08,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], 0.5, v[8:9]
// GFX1210: v_mul_u64_e32 v[4:5], 0.5, v[8:9]       ; encoding: [0xf0,0x10,0x08,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], -4.0, v[8:9]
// GFX1210: v_mul_u64_e32 v[4:5], -4.0, v[8:9]      ; encoding: [0xf7,0x10,0x08,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], v[2:3], v[254:255]
// GFX1210: v_mul_u64_e32 v[4:5], v[2:3], v[254:255] ; encoding: [0x02,0xfd,0x09,0x54]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], v[2:3], vcc
// GFX1210: v_mul_u64_e64 v[4:5], v[2:3], vcc       ; encoding: [0x04,0x00,0x2a,0xd5,0x02,0xd5,0x00,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], v[2:3], exec
// GFX1210: v_mul_u64_e64 v[4:5], v[2:3], exec      ; encoding: [0x04,0x00,0x2a,0xd5,0x02,0xfd,0x00,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], v[2:3], 0
// GFX1210: v_mul_u64_e64 v[4:5], v[2:3], 0         ; encoding: [0x04,0x00,0x2a,0xd5,0x02,0x01,0x01,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], v[2:3], -1
// GFX1210: v_mul_u64_e64 v[4:5], v[2:3], -1        ; encoding: [0x04,0x00,0x2a,0xd5,0x02,0x83,0x01,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], v[2:3], 0.5
// GFX1210: v_mul_u64_e64 v[4:5], v[2:3], 0.5       ; encoding: [0x04,0x00,0x2a,0xd5,0x02,0xe1,0x01,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_mul_u64 v[4:5], v[2:3], -4.0
// GFX1210: v_mul_u64_e64 v[4:5], v[2:3], -4.0      ; encoding: [0x04,0x00,0x2a,0xd5,0x02,0xef,0x01,0x00]
// GFX1200-ERR: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU
