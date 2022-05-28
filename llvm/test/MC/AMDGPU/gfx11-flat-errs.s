// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga 2>&1 %s | FileCheck -check-prefix=VI-GFX9_10-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 2>&1 %s | FileCheck -check-prefix=VI-GFX9_10-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 2>&1 %s | FileCheck --check-prefix=VI-GFX9_10-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1030 2>&1 %s | FileCheck --check-prefix=VI-GFX9_10-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1100 2>&1 %s | FileCheck --check-prefix=GFX11-ERR --implicit-check-not=error: %s

// FLAT

flat_load_u8 v1, v[3:4]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_i8 v1, v[3:4]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_u16 v1, v[3:4]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_i16 v1, v[3:4]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_b16 v1, v[3:4]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4] offset:-1
// GFX11-ERR: error: expected a 12-bit unsigned offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4] offset:2047
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4] offset:4096
// GFX11-ERR: error: expected a 12-bit unsigned offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4] offset:4 glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4] offset:4 glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4] offset:4 glc slc dlc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b64 v[1:2], v[3:4]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b96 v[1:3], v[5:6]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b128 v[1:4], v[5:6]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_i8 v1, v[3:4]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_hi_i8 v1, v[3:4]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b8 v[3:4], v1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b16 v[3:4], v1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b32 v[3:4], v1 offset:16
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b32 v[3:4], v1, off
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b32 v[3:4], v1, s[0:1]
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b32 v[3:4], v1, s0
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4], off
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4], s[0:1]
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4], s0
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_b32 v1, v[3:4], exec_hi
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b32 v[3:4], v1, exec_hi
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b64 v[1:2], v[3:4]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b96 v[1:2], v[3:5]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_b128 v[1:2], v[3:6]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3 offset:2047
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v[1:2], v3 offset:2047 glc
// GFX11-ERR: error: instruction must not use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3 offset:2047 glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3 offset:2047 glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3 glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3 glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3 slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b32 v0, v[1:2], v3 offset:2047 slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b64 v[1:2], v[3:4] offset:2047 glc
// GFX11-ERR: error: instruction must not use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b64 v[1:2], v[3:4] glc
// GFX11-ERR: error: instruction must not use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b64 v[1:2], v[3:4], v[5:6] offset:2047 glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b64 v[1:2], v[3:4], v[5:6] offset:2047 glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b64 v[1:2], v[3:4], v[5:6] glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_swap_b64 v[1:2], v[3:4], v[5:6] glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_add_u32 v[3:4], v5 slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_add_u32 v2, v[3:4], v5 slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_add_u32 v1, v[3:4], v5 offset:8 slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4] offset:2047
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v[1:2], v[3:4] offset:2047 glc
// GFX11-ERR: error: instruction must not use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4] offset:2047 glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4] offset:2047 glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4] glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4] glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4]
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4] slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b32 v0, v[1:2], v[3:4] offset:2047 slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b64 v[1:2], v[3:4] offset:2047
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b64 v[1:2], v[3:4] offset:2047 glc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b64 v[1:2], v[3:4] glc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b64 v[1:2], v[3:4], v[5:8] offset:2047 glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b64 v[1:2], v[3:4], v[5:8] offset:2047 glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b64 v[1:2], v[3:4], v[5:8] glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_atomic_cmpswap_b64 v[1:2], v[3:4], v[5:8] glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_u8 v1, v[3:4]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_hi_u8 v1, v[3:4]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_i8 v1, v[3:4]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_hi_i8 v1, v[3:4]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_b16 v1, v[3:4]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_load_d16_hi_b16 v1, v[3:4]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_d16_hi_b8 v[3:4], v1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

flat_store_d16_hi_b16 v[3:4], v1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

// GLOBAL No saddr

global_load_u8 v1, v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_i8 v1, v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_u16 v1, v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_i16 v1, v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_b16 v1, v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off offset:-1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off offset:2047
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off offset:2048
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off offset:4096
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: expected a 13-bit signed offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off offset:4 glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off offset:4 glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off offset:4 glc slc dlc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b64 v[1:2], v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b96 v[1:3], v[5:6], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b128 v[1:4], v[5:6], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_i8 v1, v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_hi_i8 v1, v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b8 v[3:4], v1, off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b16 v[3:4], v1, off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v[3:4], v1, off offset:16
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v[3:4], v1
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v[3:4], v1, s[0:1]
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v[3:4], v1, s0
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], off, s[0:1]
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], s0
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v[3:4], exec_hi
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v[3:4], v1, exec_hi
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b64 v[1:2], v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b96 v[1:2], v[3:5], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b128 v[1:2], v[3:6], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, off offset:2047
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v[1:2], v3 offset:2047 glc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: not a valid operand
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v[1:2], v3 glc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, off offset:2047 glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, off offset:2047 glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, off glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, off glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, off
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, off slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, off offset:2047 slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b64 v[1:2], v[3:4], v[5:6], off offset:2047 glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b64 v[1:2], v[3:4], v[5:6], off offset:2047 glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b64 v[1:2], v[3:4], v[5:6], off glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b64 v[1:2], v[3:4], v[5:6], off glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_add_u32 v2, v[3:4], off, v5 slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_add_u32 v1, v[3:4], off, v5 offset:8 slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v[1:2], v[3:4], off offset:2047
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v[1:2], v[3:4], off offset:2047 glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v[1:2], v[3:4], off offset:2047 glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v[1:2], v[3:4], off glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v[1:2], v[3:4], off glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v[1:2], off, v[3:4]
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v[1:2], v[3:4], off slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v[1:2], v[3:4], off offset:2047 slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b64 v[1:2], v[3:4], off offset:2047
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b64 v[1:2], v[3:4], v[5:8], off offset:2047 glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b64 v[1:2], v[3:4], v[5:8], off offset:2047 glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b64 v[1:2], v[3:4], v[5:8], off glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b64 v[1:2], v[3:4], v[5:8], off glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_u8 v1, v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_hi_u8 v1, v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_i8 v1, v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_hi_i8 v1, v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_b16 v1, v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_hi_b16 v1, v[3:4], off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_d16_hi_b8 v[3:4], v1, off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_d16_hi_b16 v[3:4], v1, off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_addtid_b32 v1, off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

// GLOBAL With saddr

global_load_u8 v1, v3, s2
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_u8 v1, v3, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_i8 v1, v3, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_u16 v1, v3, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_i16 v1, v3, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_b16 v1, v3, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3] offset:-1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3] offset:-4097
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: expected a 13-bit signed offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3] offset:2047
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3] offset:2048
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3] offset:4096
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: expected a 13-bit signed offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3] offset:4 glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3] offset:4 glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3] offset:4 glc slc dlc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b64 v[1:2], v3, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b96 v[1:3], v5, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b128 v[1:4], v5, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_i8 v1, v3, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_hi_i8 v1, v3, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b8 v3, v1, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b16 v3, v1, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v3, v1, s[2:3] offset:16
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v3, v1
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: too few operands for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v3, v1, s[0:1]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v3, v1, s0
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s[2:3], s[0:1]
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, s0
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_b32 v1, v3, exec_hi
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b32 v3, v1, exec_hi
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b64 v1, v[2:3], s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b96 v1, v[3:5], s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_b128 v1, v[3:6], s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, s[2:3] offset:2047
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v[1:2], v3 offset:2047 glc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: not a valid operand
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v[1:2], v3 glc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v1, v3, s[2:3] offset:2047 glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v1, v3, s[2:3] offset:2047 glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v1, v3, s[2:3] glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v1, v3, s[2:3] glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v1, v3, s[2:3]
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v1, v3, s[2:3] slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: instruction must use glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b32 v0, v[1:2], v3, s[2:3] offset:2047 slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b64 v[1:2], v3, v[5:6], s[2:3] offset:2047 glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b64 v[1:2], v3, v[5:6], s[2:3] offset:2047 glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b64 v[1:2], v3, v[5:6], s[2:3] glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_swap_b64 v[1:2], v3, v[5:6], s[2:3] glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_add_u32 v2, v3, s[2:3], v5 slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_add_u32 v1, v3, s[2:3], v5 offset:8 slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, v3, s[2:3] offset:2047
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, v[2:3], s[2:3] offset:2047 glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, v[2:3], s[2:3] offset:2047 glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, v[2:3], s[2:3] glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, v[2:3], s[2:3] glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, s[2:3], v3
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, v3, s[2:3] slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b32 v0, v1, v3, s[2:3] offset:2047 slc
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b64 v[1:2], v3, s[2:3] offset:2047
// GFX11-ERR: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b64 v[1:2], v3, v[5:8], s[2:3] offset:2047 glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b64 v[1:2], v3, v[5:8], s[2:3] offset:2047 glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b64 v[1:2], v3, v[5:8], s[2:3] glc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_atomic_cmpswap_b64 v[1:2], v3, v[5:8], s[2:3] glc slc
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_u8 v1, v3, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_hi_u8 v1, v3, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_i8 v1, v3, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_hi_i8 v1, v3, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_b16 v1, v3, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_d16_hi_b16 v1, v3, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_d16_hi_b8 v3, v1, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_store_d16_hi_b16 v3, v1, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

global_load_addtid_b32 v1, s[2:3]
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

// SCRATCH

scratch_load_u8 v1, v2, s1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_i8 v1, v2, s1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_u16 v1, v2, s1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_i16 v1, v2, s1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, v2, s1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b64 v[1:2], v2, s1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b96 v[1:3], v2, s1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b128 v[1:4], v2, s1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b8 v1, v2, s3
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b16 v1, v2, s3
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 v1, v2, s3
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b64 v1, v[2:3], s3
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b96 v1, v[2:4], s3
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b128 v1, v[2:5], s3
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_d16_u8 v1, v2, s1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_d16_hi_u8 v1, v2, s1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_d16_i8 v1, v2, s1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_d16_hi_i8 v1, v2, s1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_d16_b16 v1, v2, s1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_d16_hi_b16 v1, v2, s1
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_d16_hi_b8 v1, v2, s3
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_d16_hi_b16 v1, v2, s3
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, v2, s1 offset:2047
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, v2, off offset:2047
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, off, s1 offset:2047
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, off, off offset:2047
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, off, off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 v1, v2, s3 offset:2047
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 v1, v2, off offset:2047
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 off, v2, s3 offset:2047
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 off, v2, off offset:2047
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, v2, s1 offset:4095
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, v2, s1 offset:-4096
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 v1, v2, s1 offset:4095
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 v1, v2, s1 offset:-4096
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, v2, s1 offset:4096
// GFX11-ERR: error: expected a 13-bit signed offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_load_b32 v1, v2, s1 offset:-4097
// GFX11-ERR: error: expected a 13-bit signed offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 v1, v2, s1 offset:4096
// GFX11-ERR: error: expected a 13-bit signed offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 v1, v2, s1 offset:-4097
// GFX11-ERR: error: expected a 13-bit signed offset
// VI-GFX9_10-ERR: error: instruction not supported on this GPU

scratch_store_b32 off, v2, off
// VI-GFX9_10-ERR: error: instruction not supported on this GPU
