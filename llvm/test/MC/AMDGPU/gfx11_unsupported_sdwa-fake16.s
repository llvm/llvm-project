// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=-real-true16,+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s
// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx1100 -mattr=-real-true16,-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --implicit-check-not=error: %s

v_floor_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported

v_ceil_f16_sdwa v255, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: sdwa variant of this instruction is not supported
