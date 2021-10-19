// RUN: not llvm-mc -arch=amdgcn -mcpu=tonga %s 2>&1 | FileCheck %s --implicit-check-not=error: --strict-whitespace

//==============================================================================
// a16 modifier is not supported on this GPU

image_gather4 v[5:8], v1, s[8:15], s[12:15] dmask:0x1 a16
// CHECK: error: a16 modifier is not supported on this GPU
// CHECK-NEXT:{{^}}image_gather4 v[5:8], v1, s[8:15], s[12:15] dmask:0x1 a16
// CHECK-NEXT:{{^}}                                                      ^

image_gather4 v[5:8], v1, s[8:15], s[12:15] dmask:0x1 noa16
// CHECK: error: a16 modifier is not supported on this GPU
// CHECK-NEXT:{{^}}image_gather4 v[5:8], v1, s[8:15], s[12:15] dmask:0x1 noa16
// CHECK-NEXT:{{^}}                                                      ^

//==============================================================================
// expected a 20-bit unsigned offset

s_atc_probe 0x7, s[4:5], -1
// CHECK: error: expected a 20-bit unsigned offset
// CHECK-NEXT:{{^}}s_atc_probe 0x7, s[4:5], -1
// CHECK-NEXT:{{^}}                         ^

s_store_dword s1, s[2:3], 0xFFFFFFFFFFF00000
// CHECK: error: expected a 20-bit unsigned offset
// CHECK-NEXT:{{^}}s_store_dword s1, s[2:3], 0xFFFFFFFFFFF00000
// CHECK-NEXT:{{^}}                          ^

//==============================================================================
// flat offset modifier is not supported on this GPU

flat_atomic_add v[3:4], v5 inst_offset:8 slc
// CHECK: error: flat offset modifier is not supported on this GPU
// CHECK-NEXT:{{^}}flat_atomic_add v[3:4], v5 inst_offset:8 slc
// CHECK-NEXT:{{^}}                           ^

//==============================================================================
// image data size does not match dmask and tfe

image_gather4 v[5:6], v1, s[8:15], s[12:15] dmask:0x1 d16
// CHECK: error: image data size does not match dmask and tfe
// CHECK-NEXT:{{^}}image_gather4 v[5:6], v1, s[8:15], s[12:15] dmask:0x1 d16
// CHECK-NEXT:{{^}}^
