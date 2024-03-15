/// attribute parsing error cases.

// RUN: not llvm-mc -triple=hexagon -filetype=asm %s 2>&1 \
// RUN:   | FileCheck %s

  .attribute Tag_unknown_name, 0
// CHECK: error: attribute name not recognized: Tag_unknown_name
// CHECK-NEXT:   .attribute Tag_unknown_name

  .attribute [non_constant_expression], 0
// CHECK: error: expected numeric constant
// CHECK-NEXT:   .attribute [non_constant_expression], 0

  .attribute 42, "forty two"
// CHECK: error: expected numeric constant
// CHECK-NEXT:   .attribute 42, "forty two"

  .attribute Tag_arch, "v75"
// CHECK: error: expected numeric constant
// CHECK-NEXT:   .attribute Tag_arch, "v75"

  .attribute 0
// CHECK: :[[#@LINE-1]]:15: error: expected comma
// CHECK-NEXT:   .attribute 0
