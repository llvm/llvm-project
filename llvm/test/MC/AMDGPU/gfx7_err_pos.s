// RUN: not llvm-mc -arch=amdgcn -mcpu=bonaire %s 2>&1 | FileCheck %s --implicit-check-not=error: --strict-whitespace

//==============================================================================
// d16 modifier is not supported on this GPU

image_gather4 v[5:6], v1, s[8:15], s[12:15] dmask:0x1 d16
// CHECK: error: d16 modifier is not supported on this GPU
// CHECK-NEXT:{{^}}image_gather4 v[5:6], v1, s[8:15], s[12:15] dmask:0x1 d16
// CHECK-NEXT:{{^}}                                                      ^

//==============================================================================
// integer clamping is not supported on this GPU

v_add_co_u32 v84, s[4:5], v13, v31 clamp
// CHECK: error: integer clamping is not supported on this GPU
// CHECK-NEXT:{{^}}v_add_co_u32 v84, s[4:5], v13, v31 clamp
// CHECK-NEXT:{{^}}                                   ^

//==============================================================================
// literal operands are not supported

v_and_b32_e64 v0, 0.159154943091895317852646485335, v1
// CHECK: error: literal operands are not supported
// CHECK-NEXT:{{^}}v_and_b32_e64 v0, 0.159154943091895317852646485335, v1
// CHECK-NEXT:{{^}}                  ^
