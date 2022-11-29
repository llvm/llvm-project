// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1200 %s 2>&1 | FileCheck --implicit-check-not=error: %s

//===----------------------------------------------------------------------===//
// Unsupported instructions.
//===----------------------------------------------------------------------===//

s_waitcnt_expcnt exec_hi, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_waitcnt_lgkmcnt exec_hi, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_waitcnt_vmcnt exec_hi, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU

s_waitcnt_vscnt exec_hi, 0x1234
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction not supported on this GPU
