// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 %s 2>&1 | FileCheck --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// Unsupported instructions.
//===----------------------------------------------------------------------===//

s_subvector_loop_begin s0, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_subvector_loop_end s0, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
