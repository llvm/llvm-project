// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+pauth-lr < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+pauth-lr < %s \
// RUN:        | llvm-objdump -d --mattr=+pauth-lr - | FileCheck %s --check-prefix=CHECK-DISASS
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+pauth-lr < %s \
// RUN:        | llvm-objdump -d --mattr=-pauth-lr - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Label at address 4, so we can test that the address shows up in the
// disassembly.
  nop
label1:

  paciasppc
// CHECK-INST: paciasppc
// CHECK-DISASS: paciasppc
// CHECK-ENCODING: [0xfe,0xa3,0xc1,0xda]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: dac1a3fe <unknown>

  pacibsppc
// CHECK-INST: pacibsppc
// CHECK-DISASS: pacibsppc
// CHECK-ENCODING: [0xfe,0xa7,0xc1,0xda]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: dac1a7fe <unknown>

  pacnbiasppc
// CHECK-INST: pacnbiasppc
// CHECK-DISASS: pacnbiasppc
// CHECK-ENCODING: [0xfe,0x83,0xc1,0xda]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: dac183fe <unknown>

  pacnbibsppc
// CHECK-INST: pacnbibsppc
// CHECK-DISASS: pacnbibsppc
// CHECK-ENCODING: [0xfe,0x87,0xc1,0xda]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: dac187fe <unknown>

  autiasppc label1
// CHECK-INST: autiasppc label1
// CHECK-DISASS: autiasppc 0x4 <label1>
// CHECK-ENCODING: [0bAAA11111,A,0b100AAAAA,0xf3]
// CHECK-ENCODING: fixup A - offset: 0, value: label1, kind: fixup_aarch64_pcrel_branch16
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: f380009f <unknown>

  autibsppc label1
// CHECK-INST: autibsppc label1
// CHECK-DISASS: autibsppc 0x4 <label1>
// CHECK-ENCODING: [0bAAA11111,A,0b101AAAAA,0xf3]
// CHECK-ENCODING: fixup A - offset: 0, value: label1, kind: fixup_aarch64_pcrel_branch16
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: f3a000bf <unknown>

  autibsppc #0
// CHECK-INST: autibsppc #0
// CHECK-DISASS: autibsppc 0x1c <label1+0x18>
// CHECK-ENCODING: [0x1f,0x00,0xa0,0xf3]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: f3a0001f <unknown>

  autibsppc #-(1<<18)+4
// CHECK-INST: autibsppc #-262140
// CHECK-DISASS: autibsppc 0xfffffffffffc0024 <label1+0xfffffffffffc0020>
// CHECK-ENCODING: [0xff,0xff,0xbf,0xf3]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: f3bfffff <unknown>

  autiasppc x0
// CHECK-INST: autiasppc x0
// CHECK-DISASS: autiasppc x0
// CHECK-ENCODING: [0x1e,0x90,0xc1,0xda]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: dac1901e <unknown>

  autibsppc x1
// CHECK-INST: autibsppc x1
// CHECK-DISASS: autibsppc x1
// CHECK-ENCODING: [0x3e,0x94,0xc1,0xda]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: dac1943e <unknown>

  autiasppc xzr
// CHECK-INST: autiasppc xzr
// CHECK-DISASS: autiasppc xzr
// CHECK-ENCODING: [0xfe,0x93,0xc1,0xda]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: dac193fe <unknown>

  autibsppc xzr
// CHECK-INST: autibsppc xzr
// CHECK-DISASS: autibsppc xzr
// CHECK-ENCODING: [0xfe,0x97,0xc1,0xda]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: dac197fe <unknown>

  pacia171615
// CHECK-INST: pacia171615
// CHECK-DISASS: pacia171615
// CHECK-ENCODING: [0xfe,0x8b,0xc1,0xda]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: dac18bfe <unknown>

  pacib171615
// CHECK-INST: pacib171615
// CHECK-DISASS: pacib171615
// CHECK-ENCODING: [0xfe,0x8f,0xc1,0xda]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: dac18ffe <unknown>

  autia171615
// CHECK-INST: autia171615
// CHECK-DISASS: autia171615
// CHECK-ENCODING: [0xfe,0xbb,0xc1,0xda]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: dac1bbfe <unknown>

  autib171615
// CHECK-INST: autib171615
// CHECK-DISASS: autib171615
// CHECK-ENCODING: [0xfe,0xbf,0xc1,0xda]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: dac1bffe <unknown>

  retaasppc label1
// CHECK-INST: retaasppc label1
// CHECK-DISASS: retaasppc 0x4 <label1>
// CHECK-ENCODING: [0bAAA11111,A,0b000AAAAA,0x55]
// CHECK-ENCODING: //   fixup A - offset: 0, value: label1, kind: fixup_aarch64_pcrel_branch16
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: 5500021f <unknown>

  retabsppc label1
// CHECK-INST: retabsppc label1
// CHECK-DISASS: retabsppc 0x4 <label1>
// CHECK-ENCODING: [0bAAA11111,A,0b001AAAAA,0x55]
// CHECK-ENCODING: //   fixup A - offset: 0, value: label1, kind: fixup_aarch64_pcrel_branch16
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: 5520023f <unknown>

  retaasppc #0
// CHECK-INST: retaasppc #0
// CHECK-DISASS: retaasppc 0x4c <label1+0x48>
// CHECK-ENCODING: [0x1f,0x00,0x00,0x55]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: 5500001f <unknown>

  retaasppc #-(1<<18)+4
// CHECK-INST: retaasppc #-262140
// CHECK-DISASS: retaasppc 0xfffffffffffc0054 <label1+0xfffffffffffc0050>
// CHECK-ENCODING: [0xff,0xff,0x1f,0x55]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: 551fffff <unknown>

  retaasppc x2
// CHECK-INST: retaasppc x2
// CHECK-DISASS: retaasppc x2
// CHECK-ENCODING: [0xe2,0x0b,0x5f,0xd6]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: d65f0be2 <unknown>

  retabsppc x3
// CHECK-INST: retabsppc x3
// CHECK-DISASS: retabsppc x3
// CHECK-ENCODING: [0xe3,0x0f,0x5f,0xd6]
// CHECK-ERROR: instruction requires: pauth-lr
// CHECK-UNKNOWN: d65f0fe3 <unknown>

  pacm
// CHECK-INST: pacm
// CHECK-DISASS: pacm
// CHECK-ENCODING: [0xff,0x24,0x03,0xd5]
// CHECK-ERROR-NOT: instruction requires:
// CHECK-UNKNOWN: d50324ff hint #39
