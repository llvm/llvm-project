// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1210-ERR --implicit-check-not=error: --strict-whitespace %s

tbuffer_load_d16_format_x v4, off, s[8:11], s3 format:[BUF_FMT_8_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v255, off, s[8:11], s3 format:1 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v4, off, s[12:15], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v4, off, s[12:15], s101 format:[BUF_FMT_8_SNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v4, off, s[12:15], m0 format:2 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v4, off, s[8:11], s0 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_SNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v4, off, s[8:11], s61 format:[BUF_FMT_8_USCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v4, off, ttmp[4:7], s61 format:3 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v4, v1, s[8:11], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_USCALED] offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v4, v1, s[8:11], s3 format:[BUF_FMT_8_SSCALED] idxen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v4, v[1:2], s[8:11], s0 format:4 idxen offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_SSCALED] offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v4, off, ttmp[4:7], s3 format:[BUF_FMT_8_UINT] offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v4, off, ttmp[4:7], s3 format:5 offset:8388607 scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v4, off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UINT] offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UINT] offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_UINT] offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, off, s[8:11], s3 format:[BUF_FMT_8_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v255, off, s[8:11], s3 format:6 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, off, s[12:15], s3 format:[BUF_DATA_FORMAT_8, BUF_NUM_FORMAT_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, off, s[12:15], s101 format:[BUF_FMT_16_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, off, s[12:15], m0 format:7 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, off, s[8:11], s0 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, off, s[8:11], s61 format:[BUF_FMT_16_SNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], s61 format:8 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, v1, s[8:11], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SNORM] offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, v1, s[8:11], s3 format:[BUF_FMT_16_USCALED] idxen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, v[1:2], s[8:11], s0 format:9 idxen offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_USCALED] offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], s3 format:[BUF_FMT_16_SSCALED] offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], s3 format:10 offset:8388607 scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SSCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SSCALED] offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SSCALED] offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SSCALED] offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[254:255], off, s[8:11], s3 format:11 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], off, s[12:15], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], off, s[12:15], s101 format:[BUF_FMT_16_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], off, s[12:15], m0 format:12 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], off, s[8:11], s0 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], off, s[8:11], s61 format:[BUF_FMT_16_FLOAT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], s61 format:13 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], v1, s[8:11], s3 format:[BUF_DATA_FORMAT_16, BUF_NUM_FORMAT_FLOAT] offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], v1, s[8:11], s3 format:[BUF_FMT_8_8_UNORM] idxen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], v[1:2], s[8:11], s0 format:14 idxen offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_UNORM] offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_8_8_SNORM] offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:15 offset:8388607 scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SNORM] offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SNORM] offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SNORM] offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], off, s[8:11], s3 format:[BUF_FMT_8_8_USCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[254:255], off, s[8:11], s3 format:16 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], off, s[12:15], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_USCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], off, s[12:15], s101 format:[BUF_FMT_8_8_SSCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], off, s[12:15], m0 format:17 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], off, s[8:11], s0 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SSCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], off, s[8:11], s61 format:[BUF_FMT_8_8_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], s61 format:18 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], v1, s[8:11], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_UINT] offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], v1, s[8:11], s3 format:[BUF_FMT_8_8_SINT] idxen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], v[1:2], s[8:11], s0 format:19 idxen offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8, BUF_NUM_FORMAT_SINT] offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_32_UINT] offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:20 offset:8388607 scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_UINT] offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_UINT] offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_UINT] offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_32_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v255, off, s[8:11], s3 format:21 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[12:15], s3 format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[12:15], s101 format:[BUF_FMT_32_FLOAT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[12:15], m0 format:22 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s0 format:[BUF_DATA_FORMAT_32, BUF_NUM_FORMAT_FLOAT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s61 format:[BUF_FMT_16_16_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, ttmp[4:7], s61 format:23 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, v1, s[8:11], s3 format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_UNORM] offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, v1, s[8:11], s3 format:[BUF_FMT_16_16_SNORM] idxen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, v[1:2], s[8:11], s0 format:24 idxen offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_SNORM] offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, ttmp[4:7], s3 format:[BUF_FMT_16_16_USCALED] offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, ttmp[4:7], s3 format:25 offset:8388607 scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_USCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_USCALED] offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_USCALED] offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_USCALED] offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_SSCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[254:255], off, s[8:11], s3 format:26 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], off, s[12:15], s3 format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_SSCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], off, s[12:15], s101 format:[BUF_FMT_16_16_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], off, s[12:15], m0 format:27 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], off, s[8:11], s0 format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], off, s[8:11], s61 format:[BUF_FMT_16_16_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], s61 format:28 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], v1, s[8:11], s3 format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_SINT] offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], v1, s[8:11], s3 format:[BUF_FMT_16_16_FLOAT] idxen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], v[1:2], s[8:11], s0 format:29 idxen offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16, BUF_NUM_FORMAT_FLOAT] offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_10_11_11_FLOAT] offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], s3 format:30 offset:8388607 scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_10_11_11, BUF_NUM_FORMAT_FLOAT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_10_11_11, BUF_NUM_FORMAT_FLOAT] offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_10_11_11, BUF_NUM_FORMAT_FLOAT] offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_10_11_11, BUF_NUM_FORMAT_FLOAT] offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], off, s[8:11], s3 format:[BUF_FMT_11_11_10_FLOAT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[253:255], off, s[8:11], s3 format:31 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], off, s[12:15], s3 format:[BUF_DATA_FORMAT_11_11_10, BUF_NUM_FORMAT_FLOAT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], off, s[12:15], s101 format:[BUF_FMT_10_10_10_2_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], off, s[12:15], m0 format:32 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], off, s[8:11], s0 format:[BUF_DATA_FORMAT_10_10_10_2, BUF_NUM_FORMAT_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], off, s[8:11], s61 format:[BUF_FMT_10_10_10_2_SNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], s61 format:33 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], v1, s[8:11], s3 format:[BUF_DATA_FORMAT_10_10_10_2, BUF_NUM_FORMAT_SNORM] offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], v1, s[8:11], s3 format:[BUF_FMT_10_10_10_2_UINT] idxen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], v[1:2], s[8:11], s0 format:34 idxen offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_10_10_10_2, BUF_NUM_FORMAT_UINT] offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_FMT_10_10_10_2_SINT] offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], s3 format:35 offset:8388607 scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_10_10_10_2, BUF_NUM_FORMAT_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_10_10_10_2, BUF_NUM_FORMAT_SINT] offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_10_10_10_2, BUF_NUM_FORMAT_SINT] offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_10_10_10_2, BUF_NUM_FORMAT_SINT] offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], off, s[8:11], s3 format:[BUF_FMT_2_10_10_10_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[252:255], off, s[8:11], s3 format:36 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], off, s[12:15], s3 format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], off, s[12:15], s101 format:[BUF_FMT_2_10_10_10_SNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], off, s[12:15], m0 format:37 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], off, s[8:11], s0 format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_SNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], off, s[8:11], s61 format:[BUF_FMT_2_10_10_10_USCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], s61 format:38 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], v1, s[8:11], s3 format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_USCALED] offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], v1, s[8:11], s3 format:[BUF_FMT_2_10_10_10_SSCALED] idxen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], v[1:2], s[8:11], s0 format:39 idxen offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_SSCALED] offset:8388607 th:TH_LOAD_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_FMT_2_10_10_10_UINT] offset:8388607 th:TH_LOAD_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], s3 format:40 offset:8388607 scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_UINT] offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_UINT] offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_UINT] offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, off, s[8:11], s3 format:[BUF_FMT_2_10_10_10_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v255, off, s[8:11], s3 format:41 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, off, s[12:15], s3 format:[BUF_DATA_FORMAT_2_10_10_10, BUF_NUM_FORMAT_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, off, s[12:15], s101 format:[BUF_FMT_8_8_8_8_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, off, s[12:15], m0 format:42 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, off, s[8:11], s0 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, off, s[8:11], s61 format:[BUF_FMT_8_8_8_8_SNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, off, ttmp[4:7], s61 format:43 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, v1, s[8:11], s3 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SNORM] offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, v1, s[8:11], s3 format:[BUF_FMT_8_8_8_8_USCALED] idxen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, v[1:2], s[8:11], s0 format:44 idxen offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_USCALED] offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, off, ttmp[4:7], s3 format:[BUF_FMT_8_8_8_8_SSCALED] offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, off, ttmp[4:7], s3 format:45 offset:8388607 scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SSCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SSCALED] offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SSCALED] offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_x v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SSCALED] offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, off, s[8:11], s3 format:[BUF_FMT_8_8_8_8_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v255, off, s[8:11], s3 format:46 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, off, s[12:15], s3 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, off, s[12:15], s101 format:[BUF_FMT_8_8_8_8_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, off, s[12:15], m0 format:47 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, off, s[8:11], s0 format:[BUF_DATA_FORMAT_8_8_8_8, BUF_NUM_FORMAT_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, off, s[8:11], s61 format:[BUF_FMT_32_32_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], s61 format:48 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, v1, s[8:11], s3 format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_UINT] offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, v1, s[8:11], s3 format:[BUF_FMT_32_32_SINT] idxen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, v[1:2], s[8:11], s0 format:49 idxen offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_SINT] offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], s3 format:[BUF_FMT_32_32_FLOAT] offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], s3 format:50 offset:8388607 scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xy v4, off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_16_16_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[254:255], off, s[8:11], s3 format:51 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], off, s[12:15], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], off, s[12:15], s101 format:[BUF_FMT_16_16_16_16_SNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], off, s[12:15], m0 format:52 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], off, s[8:11], s0 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_SNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], off, s[8:11], s61 format:[BUF_FMT_16_16_16_16_USCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], s61 format:53 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], v1, s[8:11], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_USCALED] offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], v1, s[8:11], s3 format:[BUF_FMT_16_16_16_16_SSCALED] idxen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], v[1:2], s[8:11], s0 format:54 idxen offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_SSCALED] offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_16_16_16_16_UINT] offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:55 offset:8388607 scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UINT] offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UINT] offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyz v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_UINT] offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_16_16_16_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[254:255], off, s[8:11], s3 format:56 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], off, s[12:15], s3 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], off, s[12:15], s101 format:[BUF_FMT_16_16_16_16_FLOAT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], off, s[12:15], m0 format:57 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], off, s[8:11], s0 format:[BUF_DATA_FORMAT_16_16_16_16, BUF_NUM_FORMAT_FLOAT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], off, s[8:11], s61 format:[BUF_FMT_32_32_32_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], s61 format:58 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], v1, s[8:11], s3 format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_UINT] offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], v1, s[8:11], s3 format:[BUF_FMT_32_32_32_SINT] idxen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], v[1:2], s[8:11], s0 format:59 idxen offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_SINT] offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_32_32_32_FLOAT] offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:60 offset:8388607 scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], 0 format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_d16_format_xyzw v[4:5], off, ttmp[4:7], s3 format:[BUF_DATA_FORMAT_32_32_32, BUF_NUM_FORMAT_FLOAT] offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, off, s[8:11], s3 format:[BUF_FMT_32_32_32_32_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v255, off, s[8:11], s3 format:61 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, off, s[12:15], s3 format:[BUF_DATA_FORMAT_32_32_32_32, BUF_NUM_FORMAT_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, off, s[12:15], s101 format:[BUF_FMT_32_32_32_32_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, off, s[12:15], m0 format:62 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, off, s[8:11], s0 format:[BUF_DATA_FORMAT_32_32_32_32, BUF_NUM_FORMAT_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, off, s[8:11], s61 format:[BUF_FMT_32_32_32_32_FLOAT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, off, ttmp[4:7], s61 format:63 offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, v1, s[8:11], s3 format:[BUF_DATA_FORMAT_32_32_32_32, BUF_NUM_FORMAT_FLOAT] offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, v1, s[8:11], s3 format:[BUF_FMT_8_UNORM] idxen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, v[1:2], s[8:11], s0 format:[BUF_FMT_8_SNORM] idxen offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, off, ttmp[4:7], s3 format:[BUF_FMT_8_USCALED] offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, off, ttmp[4:7], s3 format:[BUF_FMT_8_SSCALED] offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, off, ttmp[4:7], s3 format:[BUF_FMT_8_UINT] offset:8388607 scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, off, ttmp[4:7], 0 format:[BUF_FMT_8_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, off, ttmp[4:7], s3 format:[BUF_FMT_8_SINT] offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, off, ttmp[4:7], s3 format:[BUF_FMT_8_SINT] offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_x v4, off, ttmp[4:7], s3 format:[BUF_FMT_8_SINT] offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], off, s[8:11], s3 format:[BUF_FMT_16_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[254:255], off, s[8:11], s3 format:[BUF_FMT_16_SNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], off, s[12:15], s3 format:[BUF_FMT_16_USCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], off, s[12:15], s101 format:[BUF_FMT_16_SSCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], off, s[12:15], m0 format:[BUF_FMT_16_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], off, s[8:11], s0 format:[BUF_FMT_16_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], off, s[8:11], s61 format:[BUF_FMT_16_FLOAT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], s61 format:[BUF_FMT_8_8_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], v1, s[8:11], s3 format:[BUF_FMT_8_8_SNORM] offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], v1, s[8:11], s3 format:[BUF_FMT_8_8_USCALED] idxen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], v[1:2], s[8:11], s0 format:[BUF_FMT_8_8_SSCALED] idxen offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_8_8_UINT] offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_8_8_SINT] offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_32_UINT] offset:8388607 scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], 0 format:[BUF_FMT_32_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_32_SINT] offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_32_SINT] offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xy v[4:5], off, ttmp[4:7], s3 format:[BUF_FMT_32_SINT] offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], off, s[8:11], s3 format:[BUF_FMT_32_FLOAT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[253:255], off, s[8:11], s3 format:[BUF_FMT_16_16_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], off, s[12:15], s3 format:[BUF_FMT_16_16_SNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], off, s[12:15], s101 format:[BUF_FMT_16_16_USCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], off, s[12:15], m0 format:[BUF_FMT_16_16_SSCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], off, s[8:11], s0 format:[BUF_FMT_16_16_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], off, s[8:11], s61 format:[BUF_FMT_16_16_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], s61 format:[BUF_FMT_16_16_FLOAT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], v1, s[8:11], s3 format:[BUF_FMT_10_11_11_FLOAT] offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], v1, s[8:11], s3 format:[BUF_FMT_11_11_10_FLOAT] idxen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], v[1:2], s[8:11], s0 format:[BUF_FMT_10_10_10_2_UNORM] idxen offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_FMT_10_10_10_2_SNORM] offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_FMT_10_10_10_2_UINT] offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_FMT_10_10_10_2_SINT]  offset:8388607 scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], 0 format:[BUF_FMT_2_10_10_10_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_FMT_2_10_10_10_UNORM] offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_FMT_2_10_10_10_UNORM] offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyz v[4:6], off, ttmp[4:7], s3 format:[BUF_FMT_2_10_10_10_UNORM] offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], off, s[8:11], s3 format:[BUF_FMT_2_10_10_10_SNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[252:255], off, s[8:11], s3 format:[BUF_FMT_2_10_10_10_USCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], off, s[12:15], s3 format:[BUF_FMT_2_10_10_10_SSCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], off, s[12:15], s101 format:[BUF_FMT_2_10_10_10_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], off, s[12:15], m0 format:[BUF_FMT_2_10_10_10_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], off, s[8:11], s0 format:[BUF_FMT_8_8_8_8_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], off, s[8:11], s61 format:[BUF_FMT_8_8_8_8_SNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], s61 format:[BUF_FMT_8_8_8_8_USCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], v1, s[8:11], s3 format:[BUF_FMT_8_8_8_8_SSCALED] offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], v1, s[8:11], s3 format:[BUF_FMT_8_8_8_8_UINT] idxen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], v[1:2], s[8:11], s0 format:[BUF_FMT_8_8_8_8_SINT] idxen offen offset:52
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_FMT_32_32_UINT] offset:8388607 th:TH_STORE_NT_HT scope:SCOPE_DEV
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_FMT_32_32_SINT] offset:8388607 th:TH_STORE_BYPASS scope:SCOPE_SYS
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_FMT_32_32_FLOAT] offset:8388607 scope:SCOPE_SE
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], 0 format:[BUF_FMT_16_16_16_16_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_FMT_16_16_16_16_UNORM] offset:8388607 glc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_FMT_16_16_16_16_UNORM] offset:8388607 slc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_store_format_xyzw v[4:7], off, ttmp[4:7], s3 format:[BUF_FMT_16_16_16_16_UNORM] offset:8388607 dlc
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_10_11_11_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_10_11_11_SNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_10_11_11_USCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_10_11_11_SSCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_10_11_11_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_10_11_11_SINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_11_11_10_UNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_11_11_10_SNORM] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_11_11_10_USCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_11_11_10_SSCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_11_11_10_UINT] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_10_10_10_2_USCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

tbuffer_load_format_x v4, off, s[8:11], s3 format:[BUF_FMT_10_10_10_2_SSCALED] offset:8388607
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

