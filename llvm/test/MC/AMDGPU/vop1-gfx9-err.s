// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck -check-prefixes=GCN,GFX9 --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck -check-prefixes=GCN,VI --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=hawaii %s 2>&1 | FileCheck -check-prefixes=GCN,CI --implicit-check-not=error: %s

v_swap_b32 v1, 1
// CI: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX9: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction
// VI: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swap_b32 v1, s0
// CI: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX9: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction
// VI: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swap_b32_e64 v1, v2
// GFX9: :[[@LINE-1]]:1: error: e64 variant of this instruction is not supported
// CI: :[[@LINE-2]]:1: error: instruction not supported on this GPU
// VI: :[[@LINE-3]]:1: error: instruction not supported on this GPU

v_swap_b32 v1, v2, v1
// CI: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX9: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction
// VI: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swap_b32 v1, v2, v2
// CI: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX9: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction
// VI: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swap_b32 v1, v2, v2, v2
// CI: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
// GFX9: :[[@LINE-2]]:{{[0-9]+}}: error: invalid operand for instruction
// VI: :[[@LINE-3]]:{{[0-9]+}}: error: instruction not supported on this GPU

v_swap_codegen_pseudo_b32 v1, v2
// GCN: :[[@LINE-1]]:1: error: invalid instruction
