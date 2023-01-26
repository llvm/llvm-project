// RUN: llvm-mc -triple aarch64 -show-encoding -mattr=+rasv2 < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64 -show-encoding -mattr=+v8.9a < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64 -show-encoding -mattr=+v9.4a < %s | FileCheck %s

// RUN: not llvm-mc -triple aarch64-none-linux-gnu               < %s 2>&1 | FileCheck --check-prefix=ERROR-NO-RAS %s

mrs x0, ERXGSR_EL1
// CHECK: mrs x0, ERXGSR_EL1                  // encoding: [0x40,0x53,0x38,0xd5]
// ERROR-NO-RAS: [[@LINE-2]]:9: error: expected readable system register
