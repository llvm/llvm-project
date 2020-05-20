// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.3a -o - %s 2>&1 | \
// RUN: FileCheck --check-prefixes=CHECK,ALL %s

// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding %s -o - > %t.1 2>%t.2
// RUN: FileCheck --check-prefixes=NO83,ALL %s < %t.1
// RUN: FileCheck --check-prefix=CHECK-REQ %s < %t.2

// ALL: .text
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
// ALL-EMPTY:
// ALL-EMPTY:
// CHECK-NEXT: mrs x0, APIAKeyLo_EL1     // encoding: [0x00,0x21,0x38,0xd5]
// CHECK-NEXT: mrs x0, APIAKeyHi_EL1     // encoding: [0x20,0x21,0x38,0xd5]
// CHECK-NEXT: mrs x0, APIBKeyLo_EL1     // encoding: [0x40,0x21,0x38,0xd5]
// CHECK-NEXT: mrs x0, APIBKeyHi_EL1     // encoding: [0x60,0x21,0x38,0xd5]
// CHECK-NEXT: mrs x0, APDAKeyLo_EL1     // encoding: [0x00,0x22,0x38,0xd5]
// CHECK-NEXT: mrs x0, APDAKeyHi_EL1     // encoding: [0x20,0x22,0x38,0xd5]
// CHECK-NEXT: mrs x0, APDBKeyLo_EL1     // encoding: [0x40,0x22,0x38,0xd5]
// CHECK-NEXT: mrs x0, APDBKeyHi_EL1     // encoding: [0x60,0x22,0x38,0xd5]
// CHECK-NEXT: mrs x0, APGAKeyLo_EL1     // encoding: [0x00,0x23,0x38,0xd5]
// CHECK-NEXT: mrs x0, APGAKeyHi_EL1     // encoding: [0x20,0x23,0x38,0xd5]

// CHECK-REQ: error: expected readable system register
// CHECK-REQ-NEXT: mrs x0, apiakeylo_el1
// CHECK-REQ: error: expected readable system register
// CHECK-REQ-NEXT: mrs x0, apiakeyhi_el1
// CHECK-REQ: error: expected readable system register
// CHECK-REQ-NEXT: mrs x0, apibkeylo_el1
// CHECK-REQ: error: expected readable system register
// CHECK-REQ-NEXT: mrs x0, apibkeyhi_el1
// CHECK-REQ: error: expected readable system register
// CHECK-REQ-NEXT: mrs x0, apdakeylo_el1
// CHECK-REQ: error: expected readable system register
// CHECK-REQ-NEXT: mrs x0, apdakeyhi_el1
// CHECK-REQ: error: expected readable system register
// CHECK-REQ-NEXT: mrs x0, apdbkeylo_el1
// CHECK-REQ: error: expected readable system register
// CHECK-REQ-NEXT: mrs x0, apdbkeyhi_el1
// CHECK-REQ: error: expected readable system register
// CHECK-REQ-NEXT: mrs x0, apgakeylo_el1
// CHECK-REQ: error: expected readable system register
// CHECK-REQ-NEXT: mrs x0, apgakeyhi_el1

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
// ALL-EMPTY:
// ALL-EMPTY:
// CHECK-NEXT: msr APIAKeyLo_EL1, x0     // encoding: [0x00,0x21,0x18,0xd5]
// CHECK-NEXT: msr APIAKeyHi_EL1, x0     // encoding: [0x20,0x21,0x18,0xd5]
// CHECK-NEXT: msr APIBKeyLo_EL1, x0     // encoding: [0x40,0x21,0x18,0xd5]
// CHECK-NEXT: msr APIBKeyHi_EL1, x0     // encoding: [0x60,0x21,0x18,0xd5]
// CHECK-NEXT: msr APDAKeyLo_EL1, x0     // encoding: [0x00,0x22,0x18,0xd5]
// CHECK-NEXT: msr APDAKeyHi_EL1, x0     // encoding: [0x20,0x22,0x18,0xd5]
// CHECK-NEXT: msr APDBKeyLo_EL1, x0     // encoding: [0x40,0x22,0x18,0xd5]
// CHECK-NEXT: msr APDBKeyHi_EL1, x0     // encoding: [0x60,0x22,0x18,0xd5]
// CHECK-NEXT: msr APGAKeyLo_EL1, x0     // encoding: [0x00,0x23,0x18,0xd5]
// CHECK-NEXT: msr APGAKeyHi_EL1, x0     // encoding: [0x20,0x23,0x18,0xd5]

// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ-NEXT:  msr apiakeylo_el1, x0
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ-NEXT:  msr apiakeyhi_el1, x0
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ-NEXT:  msr apibkeylo_el1, x0
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ-NEXT:  msr apibkeyhi_el1, x0
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ-NEXT:  msr apdakeylo_el1, x0
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ-NEXT:  msr apdakeyhi_el1, x0
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ-NEXT:  msr apdbkeylo_el1, x0
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ-NEXT:  msr apdbkeyhi_el1, x0
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ-NEXT:  msr apgakeylo_el1, x0
// CHECK-REQ: error: expected writable system register or pstate
// CHECK-REQ-NEXT:  msr apgakeyhi_el1, x0

// ALL-EMPTY:
// ALL-EMPTY:
  paciasp
// CHECK-NEXT: paciasp        // encoding: [0x3f,0x23,0x03,0xd5]
// NO83-NEXT: hint #25        // encoding: [0x3f,0x23,0x03,0xd5]
  autiasp
// CHECK-NEXT: autiasp        // encoding: [0xbf,0x23,0x03,0xd5]
// NO83-NEXT: hint #29        // encoding: [0xbf,0x23,0x03,0xd5]
  paciaz
// CHECK-NEXT: paciaz         // encoding: [0x1f,0x23,0x03,0xd5]
// NO83-NEXT: hint #24        // encoding: [0x1f,0x23,0x03,0xd5]
  autiaz
// CHECK-NEXT: autiaz         // encoding: [0x9f,0x23,0x03,0xd5]
// NO83-NEXT: hint #28        // encoding: [0x9f,0x23,0x03,0xd5]
  pacia1716
// CHECK-NEXT: pacia1716      // encoding: [0x1f,0x21,0x03,0xd5]
// NO83-NEXT: hint #8         // encoding: [0x1f,0x21,0x03,0xd5]
  autia1716
// CHECK-NEXT: autia1716      // encoding: [0x9f,0x21,0x03,0xd5]
// NO83-NEXT: hint #12        // encoding: [0x9f,0x21,0x03,0xd5]
  pacibsp
// CHECK-NEXT: pacibsp        // encoding: [0x7f,0x23,0x03,0xd5]
// NO83-NEXT: hint #27        // encoding: [0x7f,0x23,0x03,0xd5]
  autibsp
// CHECK-NEXT: autibsp        // encoding: [0xff,0x23,0x03,0xd5]
// NO83-NEXT: hint #31        // encoding: [0xff,0x23,0x03,0xd5]
  pacibz
// CHECK-NEXT: pacibz         // encoding: [0x5f,0x23,0x03,0xd5]
// NO83-NEXT: hint #26        // encoding: [0x5f,0x23,0x03,0xd5]
  autibz
// CHECK-NEXT: autibz         // encoding: [0xdf,0x23,0x03,0xd5]
// NO83-NEXT: hint #30        // encoding: [0xdf,0x23,0x03,0xd5]
  pacib1716
// CHECK-NEXT: pacib1716      // encoding: [0x5f,0x21,0x03,0xd5]
// NO83-NEXT: hint #10        // encoding: [0x5f,0x21,0x03,0xd5]
  autib1716
// CHECK-NEXT: autib1716      // encoding: [0xdf,0x21,0x03,0xd5]
// NO83-NEXT: hint #14        // encoding: [0xdf,0x21,0x03,0xd5]
  xpaclri
// CHECK-NEXT: xpaclri        // encoding: [0xff,0x20,0x03,0xd5]
// NO83-NEXT: hint #7         // encoding: [0xff,0x20,0x03,0xd5]

// ALL-EMPTY:
  pacia x0, x1
// CHECK-NEXT: pacia x0, x1     // encoding: [0x20,0x00,0xc1,0xda]
// CHECK-REQ-NEXT:      ^
// CHECK-REQ-NEXT: error: instruction requires: pa
// CHECK-REQ-NEXT: pacia x0, x1
  autia x0, x1
// CHECK-NEXT: autia x0, x1     // encoding: [0x20,0x10,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT: autia x0, x1
  pacda x0, x1
// CHECK-NEXT: pacda x0, x1     // encoding: [0x20,0x08,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  pacda x0, x1
  autda x0, x1
// CHECK-NEXT: autda x0, x1     // encoding: [0x20,0x18,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  autda x0, x1
  pacib x0, x1
// CHECK-NEXT: pacib x0, x1     // encoding: [0x20,0x04,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  pacib x0, x1
  autib x0, x1
// CHECK-NEXT: autib x0, x1     // encoding: [0x20,0x14,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  autib x0, x1
  pacdb x0, x1
// CHECK-NEXT: pacdb x0, x1     // encoding: [0x20,0x0c,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  pacdb x0, x1
  autdb x0, x1
// CHECK-NEXT: autdb x0, x1     // encoding: [0x20,0x1c,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  autdb x0, x1
  pacga x0, x1, x2
// CHECK-NEXT: pacga x0, x1, x2  // encoding: [0x20,0x30,0xc2,0x9a]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  pacga x0, x1, x2
  paciza x0
// CHECK-NEXT: paciza x0         // encoding: [0xe0,0x23,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  paciza x0
  autiza x0
// CHECK-NEXT: autiza x0         // encoding: [0xe0,0x33,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  autiza x0
  pacdza x0
// CHECK-NEXT: pacdza x0         // encoding: [0xe0,0x2b,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  pacdza x0
  autdza x0
// CHECK-NEXT: autdza x0         // encoding: [0xe0,0x3b,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  autdza x0
  pacizb x0
// CHECK-NEXT: pacizb x0         // encoding: [0xe0,0x27,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  pacizb x0
  autizb x0
// CHECK-NEXT: autizb x0         // encoding: [0xe0,0x37,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  autizb x0
  pacdzb x0
// CHECK-NEXT: pacdzb x0         // encoding: [0xe0,0x2f,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  pacdzb x0
  autdzb x0
// CHECK-NEXT: autdzb x0         // encoding: [0xe0,0x3f,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  autdzb x0
  xpaci x0
// CHECK-NEXT: xpaci x0          // encoding: [0xe0,0x43,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  xpaci x0
  xpacd x0
// CHECK-NEXT: xpacd x0          // encoding: [0xe0,0x47,0xc1,0xda]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  xpacd x0

  braa x0, x1
// CHECK-EMPTY:
// CHECK-NEXT: braa x0, x1       // encoding: [0x01,0x08,0x1f,0xd7]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  braa x0, x1
  brab x0, x1
// CHECK-NEXT: brab x0, x1       // encoding: [0x01,0x0c,0x1f,0xd7]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  brab x0, x1
  blraa x0, x1
// CHECK-NEXT: blraa x0, x1      // encoding: [0x01,0x08,0x3f,0xd7]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  blraa x0, x1
  blrab x0, x1
// CHECK-NEXT: blrab x0, x1      // encoding: [0x01,0x0c,0x3f,0xd7]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  blrab x0, x1

  braaz x0
// CHECK-EMPTY:
// CHECK-NEXT: braaz x0          // encoding: [0x1f,0x08,0x1f,0xd6]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  braaz x0
  brabz x0
// CHECK-NEXT: brabz x0          // encoding: [0x1f,0x0c,0x1f,0xd6]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  brabz x0
  blraaz x0
// CHECK-NEXT: blraaz x0         // encoding: [0x1f,0x08,0x3f,0xd6]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  blraaz x0
  blrabz x0
// CHECK-NEXT: blrabz x0         // encoding: [0x1f,0x0c,0x3f,0xd6]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  blrabz x0
  retaa
// CHECK-NEXT: retaa             // encoding: [0xff,0x0b,0x5f,0xd6]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  retaa
  retab
// CHECK-NEXT: retab             // encoding: [0xff,0x0f,0x5f,0xd6]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  retab
  eretaa
// CHECK-NEXT: eretaa            // encoding: [0xff,0x0b,0x9f,0xd6]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  eretaa
  eretab
// CHECK-NEXT: eretab            // encoding: [0xff,0x0f,0x9f,0xd6]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  eretab
  ldraa x0, [x1, 4088]
// CHECK-NEXT: ldraa x0, [x1, #4088]  // encoding: [0x20,0xf4,0x3f,0xf8]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  ldraa x0, [x1, 4088]
  ldraa x0, [x1, -4096]
// CHECK-NEXT: ldraa x0, [x1, #-4096] // encoding: [0x20,0x04,0x60,0xf8]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  ldraa x0, [x1, -4096]
  ldrab x0, [x1, 4088]
// CHECK-NEXT: ldrab x0, [x1, #4088]  // encoding: [0x20,0xf4,0xbf,0xf8]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  ldrab x0, [x1, 4088]
  ldrab x0, [x1, -4096]
// CHECK-NEXT: ldrab x0, [x1, #-4096] // encoding: [0x20,0x04,0xe0,0xf8]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  ldrab x0, [x1, -4096]
  ldraa x0, [x1, 4088]!
// CHECK-NEXT: ldraa x0, [x1, #4088]!  // encoding: [0x20,0xfc,0x3f,0xf8]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  ldraa x0, [x1, 4088]!
  ldraa x0, [x1, -4096]!
// CHECK-NEXT: ldraa x0, [x1, #-4096]! // encoding: [0x20,0x0c,0x60,0xf8]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  ldraa x0, [x1, -4096]!
  ldrab x0, [x1, 4088]!
// CHECK-NEXT: ldrab x0, [x1, #4088]!  // encoding: [0x20,0xfc,0xbf,0xf8]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  ldrab x0, [x1, 4088]!
  ldrab x0, [x1, -4096]!
// CHECK-NEXT: ldrab x0, [x1, #-4096]! // encoding: [0x20,0x0c,0xe0,0xf8]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  ldrab x0, [x1, -4096]!
  ldraa x0, [x1]
// CHECK-NEXT: ldraa x0, [x1]  // encoding: [0x20,0x04,0x20,0xf8]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  ldraa x0, [x1]
  ldrab x0, [x1]
// CHECK-NEXT: ldrab x0, [x1]  // encoding: [0x20,0x04,0xa0,0xf8]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  ldrab x0, [x1]
  ldraa x0, [x1]!
// CHECK-NEXT: ldraa x0, [x1, #0]!  // encoding: [0x20,0x0c,0x20,0xf8]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  ldraa x0, [x1]!
  ldrab x0, [x1]!
// CHECK-NEXT: ldrab x0, [x1, #0]!  // encoding: [0x20,0x0c,0xa0,0xf8]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  ldrab x0, [x1]!
  ldraa xzr, [sp, -4096]!
// CHECK-NEXT: ldraa xzr, [sp, #-4096]!  // encoding: [0xff,0x0f,0x60,0xf8]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  ldraa xzr, [sp, -4096]!
  ldrab xzr, [sp, -4096]!
// CHECK-NEXT: ldrab xzr, [sp, #-4096]!  // encoding: [0xff,0x0f,0xe0,0xf8]
// CHECK-REQ: error: instruction requires: pa
// CHECK-REQ-NEXT:  ldrab xzr, [sp, -4096]!
