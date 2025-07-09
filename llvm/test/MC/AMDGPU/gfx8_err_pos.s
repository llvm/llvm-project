// RUN: not llvm-mc -triple=amdgcn -mcpu=tonga %s 2>&1 | FileCheck %s --implicit-check-not=error: --strict-whitespace

//==============================================================================
// a16 modifier is not supported on this GPU

image_gather4 v[5:8], v1, s[8:15], s[12:15] dmask:0x1 a16
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: a16 modifier is not supported on this GPU
// CHECK-NEXT:{{^}}image_gather4 v[5:8], v1, s[8:15], s[12:15] dmask:0x1 a16
// CHECK-NEXT:{{^}}                                                      ^

image_gather4 v[5:8], v1, s[8:15], s[12:15] dmask:0x1 noa16
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: a16 modifier is not supported on this GPU
// CHECK-NEXT:{{^}}image_gather4 v[5:8], v1, s[8:15], s[12:15] dmask:0x1 noa16
// CHECK-NEXT:{{^}}                                                      ^

//==============================================================================
// expected a 20-bit unsigned offset

s_atc_probe 0x7, s[4:5], -1
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 20-bit unsigned offset
// CHECK-NEXT:{{^}}s_atc_probe 0x7, s[4:5], -1
// CHECK-NEXT:{{^}}                         ^

s_store_dword s1, s[2:3], 0xFFFFFFFFFFF00000
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: expected a 20-bit unsigned offset
// CHECK-NEXT:{{^}}s_store_dword s1, s[2:3], 0xFFFFFFFFFFF00000
// CHECK-NEXT:{{^}}                          ^

//==============================================================================
// flat offset modifier is not supported on this GPU

flat_atomic_add v[3:4], v5 inst_offset:8 slc
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: flat offset modifier is not supported on this GPU
// CHECK-NEXT:{{^}}flat_atomic_add v[3:4], v5 inst_offset:8 slc
// CHECK-NEXT:{{^}}                           ^

//==============================================================================
// image data size does not match dmask and tfe

image_gather4 v[5:6], v1, s[8:15], s[12:15] dmask:0x1 d16
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: image data size does not match dmask and tfe
// CHECK-NEXT:{{^}}image_gather4 v[5:6], v1, s[8:15], s[12:15] dmask:0x1 d16
// CHECK-NEXT:{{^}}^

//==============================================================================
// not a valid operand

v_cndmask_b32_sdwa v5, v1, sext(v2), vcc dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:BYTE_0 src1_sel:WORD_0
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// CHECK-NEXT:{{^}}v_cndmask_b32_sdwa v5, v1, sext(v2), vcc dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:BYTE_0 src1_sel:WORD_0
// CHECK-NEXT:{{^}}                           ^

v_alignbit_b32 v5, v1, v2, v3 op_sel:[1,1,1,1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// CHECK-NEXT:{{^}}v_alignbit_b32 v5, v1, v2, v3 op_sel:[1,1,1,1]
// CHECK-NEXT:{{^}}                              ^

v_alignbyte_b32 v5, v1, v2, v3 op_sel:[1,1,1,1]
// CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: not a valid operand.
// CHECK-NEXT:{{^}}v_alignbyte_b32 v5, v1, v2, v3 op_sel:[1,1,1,1]
// CHECK-NEXT:{{^}}                               ^
