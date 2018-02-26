// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.3a < %s 2>&1 | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mcpu=cortex-a75 < %s 2>&1 | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mcpu=cortex-a55 < %s 2>&1 | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.2a -mattr=+rcpc < %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+v8.2a < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-REQ %s < %t

  ldaprb w0, [x0, #0]
  ldaprh w0, [x17, #0]
  ldapr w0, [x1, #0]
  ldapr x0, [x0, #0]
  ldapr w18, [x0]
  ldapr x15, [x0]

// CHECK: ldaprb w0, [x0]    // encoding: [0x00,0xc0,0xbf,0x38]
// CHECK: ldaprh w0, [x17]   // encoding: [0x20,0xc2,0xbf,0x78]
// CHECK: ldapr w0, [x1]     // encoding: [0x20,0xc0,0xbf,0xb8]
// CHECK: ldapr x0, [x0]     // encoding: [0x00,0xc0,0xbf,0xf8]
// CHECK: ldapr w18, [x0]    // encoding: [0x12,0xc0,0xbf,0xb8]
// CHECK: ldapr x15, [x0]    // encoding: [0x0f,0xc0,0xbf,0xf8]
// CHECK-REQ: error: instruction requires: rcpc
// CHECK-REQ: error: instruction requires: rcpc
// CHECK-REQ: error: instruction requires: rcpc
// CHECK-REQ: error: instruction requires: rcpc
// CHECK-REQ: error: instruction requires: rcpc
// CHECK-REQ: error: instruction requires: rcpc
