// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+v8.3a < %s 2> %t
// RUN: FileCheck %s < %t

 msr ID_ISAR6_EL1, x0
// CHECK: error: expected writable system register or pstate
// CHECK-NEXT: msr ID_ISAR6_EL1, x0
// CHECK-NEXT:     ^

  ldraa x0, [x1, 4089]
// CHECK: error: index must be a multiple of 8 in range [-4096, 4088].
  ldraa x0, [x1, -4097]
// CHECK: error: index must be a multiple of 8 in range [-4096, 4088].
  ldraa x0, [x1, 4086]
// CHECK: error: index must be a multiple of 8 in range [-4096, 4088].
  ldrab x0, [x1, 4089]
// CHECK: error: index must be a multiple of 8 in range [-4096, 4088].
  ldrab x0, [x1, -4097]
// CHECK: error: index must be a multiple of 8 in range [-4096, 4088].
  ldrab x0, [x1, 4086]
// CHECK: error: index must be a multiple of 8 in range [-4096, 4088].
  ldraa x0, [x0, -4096]!
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unpredictable LDRA instruction, writeback base is also a destination
  ldrab x0, [x0, -4096]!
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: unpredictable LDRA instruction, writeback base is also a destination
  ldraa xzr, [xzr, -4096]!
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
  ldraa sp, [sp, -4096]!
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
  ldrab xzr, [xzr, -4096]!
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
  ldrab sp, [sp, -4096]!
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
