// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 -show-encoding %s 2>&1 | FileCheck --check-prefixes=GFX12-ERR --implicit-check-not=error: -strict-whitespace %s

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:0x7
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: expected an identifier

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_NT
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value for load instructions

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_ATOMIC_NT
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value for load instructions

image_store v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_NT
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value for store instructions

image_store v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_ATOMIC_NT
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value for store instructions

image_atomic_swap v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_NT
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value for atomic instructions

image_atomic_swap v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_NT
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value for atomic instructions

image_store v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_LU
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_RT_WB
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_NT_WB
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid th value

image_store v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_RT_WB scope:SCOPE_SYS
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: scope and th combination is not valid

image_store v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_STORE_BYPASS scope:SCOPE_DEV
// GFX12-ERR: [[@LINE-1]]:{{[0-9]+}}: error: scope and th combination is not valid

s_load_b32 s5, s[4:5], s0 offset:0x0 th:TH_LOAD_NT_RT
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid th value for SMEM instruction

s_buffer_load_b64 s[10:11], s[4:7], s0 offset:0x0 th:TH_LOAD_RT_NT
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid th value for SMEM instruction

s_load_b128 s[20:23], s[2:3], vcc_lo th:TH_LOAD_NT_HT
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: invalid th value for SMEM instruction

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D th:TH_LOAD_HT scope:SCOPE_SE th:TH_LOAD_HT
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D scope:SCOPE_SE th:TH_LOAD_HT scope:SCOPE_SE
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand

s_prefetch_inst s[14:15], 0xffffff, m0, 7
// GFX12-ERR: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 24-bit signed offset
// GFX12-ERR: s_prefetch_inst s[14:15], 0xffffff, m0, 7
// GFX12-ERR:                           ^
