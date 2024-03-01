// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1210-ERR --implicit-check-not=error: --strict-whitespace %s

buffer_load_b32 v5, v1, s[8:11], s3 offen offset:4095 scale_offset
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: scale_offset is not supported for this instruction
// GFX1210-ERR-NEXT:{{^}}buffer_load_b32 v5, v1, s[8:11], s3 offen offset:4095 scale_offset
// GFX1210-ERR-NEXT:{{^}}                                                      ^

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v255, off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[12:15], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[96:99], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, v0, s[8:11], s3 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, v0, s[8:11], s3 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], s3
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_x v5, off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v255, off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[12:15], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[96:99], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, v0, s[8:11], s3 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, v0, s[8:11], s3 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], s3
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xy v5, off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[254:255], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[12:15], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[96:99], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], v0, s[8:11], s3 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], v0, s[8:11], s3 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyz v[5:6], off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[254:255], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[12:15], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[96:99], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], v0, s[8:11], s3 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], v0, s[8:11], s3 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_format_xyzw v[5:6], off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v255, off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[12:15], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[96:99], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, v0, s[8:11], s3 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, v0, s[8:11], s3 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], s3
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_d16_hi_format_x v5, off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v255, off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[12:15], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[96:99], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, v0, s[8:11], s3 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, v0, s[8:11], s3 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], s3
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], s3 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], s3 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_x v5, off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[254:255], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[12:15], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[96:99], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], v0, s[8:11], s3 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], v0, s[8:11], s3 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], s3
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:6], off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xy v[5:7], off, s[8:11], s3 offset:8388607 glc slc dlc tfe
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[253:255], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[12:15], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[96:99], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], v0, s[8:11], s3 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], v0, s[8:11], s3 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], s3
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyz v[5:7], off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[252:255], off, s[8:11], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[12:15], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[96:99], s3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], v0, s[8:11], s3 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], v0, s[8:11], s3 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], s3
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_load_format_xyzw v[5:8], off, s[8:11], s3 offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v255, off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[16:19], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[96:99], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, v0, s[12:15], s4 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, v0, s[12:15], s4 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], s4
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_x v1, off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v255, off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[16:19], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[96:99], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, v0, s[12:15], s4 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, v0, s[12:15], s4 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], s4
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xy v1, off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[254:255], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[16:19], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[96:99], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], v0, s[12:15], s4 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], v0, s[12:15], s4 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyz v[1:2], off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[254:255], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[16:19], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[96:99], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], v0, s[12:15], s4 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], v0, s[12:15], s4 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_format_xyzw v[1:2], off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v255, off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[16:19], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[96:99], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, v0, s[12:15], s4 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, v0, s[12:15], s4 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], s4
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_d16_hi_format_x v1, off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v255, off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[16:19], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[96:99], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, v0, s[12:15], s4 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, v0, s[12:15], s4 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], s4
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], s4 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], s4 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_x v1, off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[254:255], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[16:19], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[96:99], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], v0, s[12:15], s4 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], v0, s[12:15], s4 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], s4
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xy v[1:2], off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[253:255], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[16:19], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[96:99], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], v0, s[12:15], s4 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], v0, s[12:15], s4 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], s4
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyz v[1:3], off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[252:255], off, s[12:15], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[16:19], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[96:99], s4 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], s101 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], m0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], 0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], -1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], 0.5 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], -4.0 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], v0, s[12:15], s4 idxen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], v0, s[12:15], s4 offen offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], s4
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:0
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:7
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607 glc slc dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

buffer_store_format_xyzw v[1:4], off, s[12:15], s4 offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
