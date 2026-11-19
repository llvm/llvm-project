// RUN: not llvm-mc -triple=amdgpu12.00 -filetype=null %s 2>&1 | FileCheck --check-prefixes=CHECK,GFX12 --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgpu12.50 -filetype=null %s 2>&1 | FileCheck --check-prefixes=CHECK,GFX1250 --implicit-check-not=error: %s

// The soffset field of VBUFFER instructions is 7 bits wide, so read-only
// source registers, which encode as 230-253, cannot be used there. They used
// to be accepted and truncated, e.g. src_scc (253) was encoded as m0 (125).

buffer_load_b32 v5, off, s[8:11], src_scc offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_store_b32 v1, off, s[12:15], src_scc offset:8
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

tbuffer_load_format_x v3, off, s[8:11], src_scc offset:4095
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
// GFX1250: :[[@LINE-2]]:{{[0-9]+}}: error: instruction not supported on this GPU (gfx1250): tbuffer_load_format_x

buffer_atomic_add_u32 v5, off, s[8:11], src_scc offset:4095
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode

buffer_load_b32 v5, off, s[8:11], src_flat_scratch_base_lo offset:4095
// GFX12: :[[@LINE-1]]:{{[0-9]+}}: error: src_flat_scratch_base_lo register not available on this GPU
// GFX1250: :[[@LINE-2]]:{{[0-9]+}}: error: operands are not valid for this GPU or mode
