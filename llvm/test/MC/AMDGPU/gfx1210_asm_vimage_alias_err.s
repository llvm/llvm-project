; RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1210 -show-encoding %s 2>&1 | FileCheck --check-prefix=GFX1210-ERR --implicit-check-not=error: --strict-whitespace %s

image_atomic_add v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_sub v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_smin v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_umin v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_smax v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_umax v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_inc v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_dec v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_min_num_flt v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

image_atomic_max_num_flt v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D
// GFX1210-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
