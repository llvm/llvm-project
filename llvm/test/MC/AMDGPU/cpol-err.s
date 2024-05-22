// RUN: not llvm-mc -triple=amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck %s --implicit-check-not=error: --strict-whitespace

scratch_load_ubyte v1, v2, off cpol:2
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// CHECK-NEXT:{{^}}scratch_load_ubyte v1, v2, off cpol:2
// CHECK-NEXT:{{^}}                               ^

scratch_load_ubyte v1, v2, off glc slc dlc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: dlc modifier is not supported on this GPU
// CHECK-NEXT:{{^}}scratch_load_ubyte v1, v2, off glc slc dlc
// CHECK-NEXT:{{^}}                                       ^

global_atomic_add v[3:4], v5, off slc glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must not use glc
// CHECK-NEXT:{{^}}global_atomic_add v[3:4], v5, off slc glc
// CHECK-NEXT:{{^}}                                      ^

global_atomic_add v0, v[1:2], v2, off glc 1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT:{{^}}global_atomic_add v0, v[1:2], v2, off glc 1
// CHECK-NEXT:{{^}}                                          ^

global_load_dword v3, v[0:1], off slc glc noglc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: duplicate cache policy modifier
// CHECK-NEXT:{{^}}global_load_dword v3, v[0:1], off slc glc noglc
// CHECK-NEXT:{{^}}                                          ^

global_load_dword v3, v[0:1], off slc glc glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: duplicate cache policy modifier
// CHECK-NEXT:{{^}}global_load_dword v3, v[0:1], off slc glc glc
// CHECK-NEXT:{{^}}                                          ^

global_load_dword v3, v[0:1], off slc noglc noglc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: duplicate cache policy modifier
// CHECK-NEXT:{{^}}global_load_dword v3, v[0:1], off slc noglc noglc
// CHECK-NEXT:{{^}}                                            ^

global_atomic_add v[3:4], v5, off slc noglc glc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: duplicate cache policy modifier
// CHECK-NEXT:{{^}}global_atomic_add v[3:4], v5, off slc noglc glc
// CHECK-NEXT:{{^}}                                            ^

s_load_dword s1, s[2:3], 0xfc glc slc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: invalid cache policy for SMEM instruction
// CHECK-NEXT:{{^}}s_load_dword s1, s[2:3], 0xfc glc slc
// CHECK-NEXT:{{^}}^
