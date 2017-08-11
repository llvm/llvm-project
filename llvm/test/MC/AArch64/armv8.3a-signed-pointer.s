// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.3a < %s 2> %t | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-REQ %s < %t

  mrs x0, apiakeylo_el1
  mrs x0, apiakeyhi_el1
  mrs x0, apibkeylo_el1
  mrs x0, apibkeyhi_el1
  mrs x0, apdakeylo_el1
  mrs x0, apdakeyhi_el1
  mrs x0, apdbkeylo_el1
  mrs x0, apdbkeyhi_el1
  mrs x0, apgakeylo_el1
  mrs x0, apgakeyhi_el1

// CHECK: mrs x0, APIAKeyLo_EL1     // encoding: [0x00,0x21,0x38,0xd5]
// CHECK: mrs x0, APIAKeyHi_EL1     // encoding: [0x20,0x21,0x38,0xd5]
// CHECK: mrs x0, APIBKeyLo_EL1     // encoding: [0x40,0x21,0x38,0xd5]
// CHECK: mrs x0, APIBKeyHi_EL1     // encoding: [0x60,0x21,0x38,0xd5]
// CHECK: mrs x0, APDAKeyLo_EL1     // encoding: [0x00,0x22,0x38,0xd5]
// CHECK: mrs x0, APDAKeyHi_EL1     // encoding: [0x20,0x22,0x38,0xd5]
// CHECK: mrs x0, APDBKeyLo_EL1     // encoding: [0x40,0x22,0x38,0xd5]
// CHECK: mrs x0, APDBKeyHi_EL1     // encoding: [0x60,0x22,0x38,0xd5]
// CHECK: mrs x0, APGAKeyLo_EL1     // encoding: [0x00,0x23,0x38,0xd5]
// CHECK: mrs x0, APGAKeyHi_EL1     // encoding: [0x20,0x23,0x38,0xd5]

// CHECK-REQ: error: expected readable system register
// CHECK-REQ: error: expected readable system register
// CHECK-REQ: error: expected readable system register
// CHECK-REQ: error: expected readable system register
// CHECK-REQ: error: expected readable system register
// CHECK-REQ: error: expected readable system register
// CHECK-REQ: error: expected readable system register
// CHECK-REQ: error: expected readable system register
// CHECK-REQ: error: expected readable system register
// CHECK-REQ: error: expected readable system register

  msr apiakeylo_el1, x0
  msr apiakeyhi_el1, x0
  msr apibkeylo_el1, x0
  msr apibkeyhi_el1, x0
  msr apdakeylo_el1, x0
  msr apdakeyhi_el1, x0
  msr apdbkeylo_el1, x0
  msr apdbkeyhi_el1, x0
  msr apgakeylo_el1, x0
  msr apgakeyhi_el1, x0

// CHECK: msr APIAKeyLo_EL1, x0     // encoding: [0x00,0x21,0x18,0xd5]
// CHECK: msr APIAKeyHi_EL1, x0     // encoding: [0x20,0x21,0x18,0xd5]
// CHECK: msr APIBKeyLo_EL1, x0     // encoding: [0x40,0x21,0x18,0xd5]
// CHECK: msr APIBKeyHi_EL1, x0     // encoding: [0x60,0x21,0x18,0xd5]
// CHECK: msr APDAKeyLo_EL1, x0     // encoding: [0x00,0x22,0x18,0xd5]
// CHECK: msr APDAKeyHi_EL1, x0     // encoding: [0x20,0x22,0x18,0xd5]
// CHECK: msr APDBKeyLo_EL1, x0     // encoding: [0x40,0x22,0x18,0xd5]
// CHECK: msr APDBKeyHi_EL1, x0     // encoding: [0x60,0x22,0x18,0xd5]
// CHECK: msr APGAKeyLo_EL1, x0     // encoding: [0x00,0x23,0x18,0xd5]
// CHECK: msr APGAKeyHi_EL1, x0     // encoding: [0x20,0x23,0x18,0xd5]

// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ: error: expected writable system register or pstate

  paciasp
// CHECK: paciasp        // encoding: [0x3f,0x23,0x03,0xd5]
// CHECK-REQ: error: instruction requires: armv8.3a
  autiasp
// CHECK: autiasp        // encoding: [0xbf,0x23,0x03,0xd5]
// CHECK-REQ: error: instruction requires: armv8.3a
  paciaz
// CHECK: paciaz         // encoding: [0x1f,0x23,0x03,0xd5]
// CHECK-REQ: error: instruction requires: armv8.3a
  autiaz
// CHECK: autiaz         // encoding: [0x9f,0x23,0x03,0xd5]
// CHECK-REQ: error: instruction requires: armv8.3a
  pacia1716
// CHECK: pacia1716      // encoding: [0x1f,0x21,0x03,0xd5]
// CHECK-REQ: error: instruction requires: armv8.3a
  autia1716
// CHECK: autia1716      // encoding: [0x9f,0x21,0x03,0xd5]
// CHECK-REQ: error: instruction requires: armv8.3a
  pacibsp
// CHECK: pacibsp        // encoding: [0x7f,0x23,0x03,0xd5]
// CHECK-REQ: error: instruction requires: armv8.3a
  autibsp
// CHECK: autibsp        // encoding: [0xff,0x23,0x03,0xd5]
// CHECK-REQ: error: instruction requires: armv8.3a
  pacibz
// CHECK: pacibz         // encoding: [0x5f,0x23,0x03,0xd5]
// CHECK-REQ: error: instruction requires: armv8.3a
  autibz
// CHECK: autibz         // encoding: [0xdf,0x23,0x03,0xd5]
// CHECK-REQ: error: instruction requires: armv8.3a
  pacib1716
// CHECK: pacib1716      // encoding: [0x5f,0x21,0x03,0xd5]
// CHECK-REQ: error: instruction requires: armv8.3a
  autib1716
// CHECK: autib1716      // encoding: [0xdf,0x21,0x03,0xd5]
// CHECK-REQ: error: instruction requires: armv8.3a
  xpaclri
// CHECK: xpaclri           // encoding: [0xff,0x20,0x03,0xd5]
// CHECK-REQ: error: instruction requires: armv8.3a

  pacia x0, x1
// CHECK: pacia x0, x1     // encoding: [0x20,0x00,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  autia x0, x1
// CHECK: autia x0, x1     // encoding: [0x20,0x10,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  pacda x0, x1
// CHECK: pacda x0, x1     // encoding: [0x20,0x08,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  autda x0, x1
// CHECK: autda x0, x1     // encoding: [0x20,0x18,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  pacib x0, x1
// CHECK: pacib x0, x1     // encoding: [0x20,0x04,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  autib x0, x1
// CHECK: autib x0, x1     // encoding: [0x20,0x14,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  pacdb x0, x1
// CHECK: pacdb x0, x1     // encoding: [0x20,0x0c,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  autdb x0, x1
// CHECK: autdb x0, x1     // encoding: [0x20,0x1c,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  pacga x0, x1, x2
// CHECK: pacga x0, x1, x2  // encoding: [0x20,0x30,0xc2,0x9a]
// CHECK-REQ: error: instruction requires: armv8.3a
  paciza x0
// CHECK: paciza x0         // encoding: [0xe0,0x23,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  autiza x0
// CHECK: autiza x0         // encoding: [0xe0,0x33,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  pacdza x0
// CHECK: pacdza x0         // encoding: [0xe0,0x2b,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  autdza x0
// CHECK: autdza x0         // encoding: [0xe0,0x3b,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  pacizb x0
// CHECK: pacizb x0         // encoding: [0xe0,0x27,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  autizb x0
// CHECK: autizb x0         // encoding: [0xe0,0x37,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  pacdzb x0
// CHECK: pacdzb x0         // encoding: [0xe0,0x2f,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  autdzb x0
// CHECK: autdzb x0         // encoding: [0xe0,0x3f,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  xpaci x0
// CHECK: xpaci x0          // encoding: [0xe0,0x43,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a
  xpacd x0
// CHECK: xpacd x0          // encoding: [0xe0,0x47,0xc1,0xda]
// CHECK-REQ: error: instruction requires: armv8.3a

  braa x0, x1
// CHECK: braa x0, x1       // encoding: [0x01,0x08,0x1f,0xd7]
// CHECK-REQ: error: instruction requires: armv8.3a
  brab x0, x1
// CHECK: brab x0, x1       // encoding: [0x01,0x0c,0x1f,0xd7]
// CHECK-REQ: error: instruction requires: armv8.3a
  blraa x0, x1
// CHECK: blraa x0, x1      // encoding: [0x01,0x08,0x3f,0xd7]
// CHECK-REQ: error: instruction requires: armv8.3a
  blrab x0, x1
// CHECK: blrab x0, x1      // encoding: [0x01,0x0c,0x3f,0xd7]
// CHECK-REQ: error: instruction requires: armv8.3a

  braaz x0
// CHECK: braaz x0          // encoding: [0x1f,0x08,0x1f,0xd6]
// CHECK-REQ: error: instruction requires: armv8.3a
  brabz x0
// CHECK: brabz x0          // encoding: [0x1f,0x0c,0x1f,0xd6]
// CHECK-REQ: error: instruction requires: armv8.3a
  blraaz x0
// CHECK: blraaz x0         // encoding: [0x1f,0x08,0x3f,0xd6]
// CHECK-REQ: error: instruction requires: armv8.3a
  blrabz x0
// CHECK: blrabz x0         // encoding: [0x1f,0x0c,0x3f,0xd6]
// CHECK-REQ: error: instruction requires: armv8.3a
  retaa
// CHECK: retaa             // encoding: [0xff,0x0b,0x5f,0xd6]
// CHECK-REQ: error: instruction requires: armv8.3a
  retab
// CHECK: retab             // encoding: [0xff,0x0f,0x5f,0xd6]
// CHECK-REQ: error: instruction requires: armv8.3a
  eretaa
// CHECK: eretaa            // encoding: [0xff,0x0b,0x9f,0xd6]
// CHECK-REQ: error: instruction requires: armv8.3a
  eretab
// CHECK: eretab            // encoding: [0xff,0x0f,0x9f,0xd6]
// CHECK-REQ: error: instruction requires: armv8.3a
  ldraa x0, [x1, 4088]
// CHECK: ldraa x0, [x1, #4088]  // encoding: [0x20,0xf4,0x3f,0xf8]
// CHECK-REQ: error: instruction requires: armv8.3a
  ldraa x0, [x1, -4096]
// CHECK: ldraa x0, [x1, #-4096] // encoding: [0x20,0x04,0x60,0xf8]
// CHECK-REQ: error: instruction requires: armv8.3a
  ldrab x0, [x1, 4088]
// CHECK: ldrab x0, [x1, #4088]  // encoding: [0x20,0xf4,0xbf,0xf8]
// CHECK-REQ: error: instruction requires: armv8.3a
  ldrab x0, [x1, -4096]
// CHECK: ldrab x0, [x1, #-4096] // encoding: [0x20,0x04,0xe0,0xf8]
// CHECK-REQ: error: instruction requires: armv8.3a
  ldraa x0, [x1, 4088]!
// CHECK: ldraa x0, [x1, #4088]!  // encoding: [0x20,0xfc,0x3f,0xf8]
// CHECK-REQ: error: instruction requires: armv8.3a
  ldraa x0, [x1, -4096]!
// CHECK: ldraa x0, [x1, #-4096]! // encoding: [0x20,0x0c,0x60,0xf8]
// CHECK-REQ: error: instruction requires: armv8.3a
  ldrab x0, [x1, 4088]!
// CHECK: ldrab x0, [x1, #4088]!  // encoding: [0x20,0xfc,0xbf,0xf8]
// CHECK-REQ: error: instruction requires: armv8.3a
  ldrab x0, [x1, -4096]!
// CHECK: ldrab x0, [x1, #-4096]! // encoding: [0x20,0x0c,0xe0,0xf8]
// CHECK-REQ: error: instruction requires: armv8.3a
  ldraa x0, [x1]
// CHECK: ldraa x0, [x1]  // encoding: [0x20,0x04,0x20,0xf8]
// CHECK-REQ: error: instruction requires: armv8.3a
  ldrab x0, [x1]
// CHECK: ldrab x0, [x1]  // encoding: [0x20,0x04,0xa0,0xf8]
// CHECK-REQ: error: instruction requires: armv8.3a
