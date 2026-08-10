// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.3a < %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-REQ %s < %t

  mrs x0, ID_ISAR6_EL1
// CHECK: mrs x0, ID_ISAR6_EL1        // encoding: [0xe0,0x02,0x38,0xd5]
// CHECK-REQ: error: expected readable system register
// CHECK-REQ-NEXT: mrs x0, ID_ISAR6_EL1
// CHECK-REQ-NEXT:         ^
