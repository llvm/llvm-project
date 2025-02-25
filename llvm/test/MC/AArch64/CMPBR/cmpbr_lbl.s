// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+cmpbr < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+cmpbr < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex --mattr=+cmpbr - | FileCheck %s --check-prefix=CHECK-DISASS
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+cmpbr < %s \
// RUN:        | llvm-objdump -d --no-print-imm-hex --mattr=-cmpbr - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// Label at address 4, so we can test that the address shows up in the
// disassembly.
  nop

lbl:

//------------------------------------------------------------------------------
// Compare & branch (Register)
//------------------------------------------------------------------------------
///
// CB<XX>
///
cbgt w5, w5, lbl
// CHECK-INST: cbgt w5, w5, lbl
// CHECK-DISASS: cbgt w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b00AAAAAA,0x05,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74050005  <unknown>

cbgt x5, x5, lbl
// CHECK-INST: cbgt x5, x5, lbl
// CHECK-DISASS: cbgt x5, x5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b00AAAAAA,0x05,0xf4]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f4053fe5

cbge x2, x2, lbl
// CHECK-INST: cbge x2, x2, lbl
// CHECK-DISASS: cbge x2, x2, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00010,0b00AAAAAA,0x22,0xf4]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f4223fc2

cbge w5, w5, lbl
// CHECK-INST: cbge w5, w5, lbl
// CHECK-DISASS: cbge w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b00AAAAAA,0x25,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74253fa5

cbhi w5, w5, lbl
// CHECK-INST: cbhi w5, w5, lbl
// CHECK-DISASS: cbhi w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b00AAAAAA,0x45,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74453f85

cbhi x5, x5, lbl
// CHECK-INST: cbhi x5, x5, lbl
// CHECK-DISASS: cbhi x5, x5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b00AAAAAA,0x45,0xf4]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f4453f65

cbhs x2, x2, lbl
// CHECK-INST: cbhs x2, x2, lbl
// CHECK-DISASS: cbhs x2, x2, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00010,0b00AAAAAA,0x62,0xf4]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f4623f42

cbhs w5, w5, lbl
// CHECK-INST: cbhs w5, w5, lbl
// CHECK-DISASS: cbhs w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b00AAAAAA,0x65,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74653f25

cbeq w5, w5, lbl
// CHECK-INST: cbeq w5, w5, lbl
// CHECK-DISASS: cbeq w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b00AAAAAA,0xc5,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74c53f05

cbeq x5, x5, lbl
// CHECK-INST: cbeq x5, x5, lbl
// CHECK-DISASS: cbeq x5, x5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b00AAAAAA,0xc5,0xf4]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f4c53ee5

cbne x2, x2, lbl
// CHECK-INST: cbne x2, x2, lbl
// CHECK-DISASS: cbne x2, x2, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00010,0b00AAAAAA,0xe2,0xf4]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f4e23ec2

cbne w5, w5, lbl
// CHECK-INST: cbne w5, w5, lbl
// CHECK-DISASS: cbne w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b00AAAAAA,0xe5,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74e53ea5

///
// CBH<XX>
///

cbhgt w5, w5, lbl
// CHECK-INST: cbhgt w5, w5, lbl
// CHECK-DISASS: cbhgt w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b11AAAAAA,0x05,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 7405fe85

cbhge w5, w5, lbl
// CHECK-INST: cbhge w5, w5, lbl
// CHECK-DISASS: cbhge w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b11AAAAAA,0x25,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 7425fe65

cbhhi w5, w5, lbl
// CHECK-INST: cbhhi w5, w5, lbl
// CHECK-DISASS: cbhhi w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b11AAAAAA,0x45,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 7445fe45

cbhhs w5, w5, lbl
// CHECK-INST: cbhhs w5, w5, lbl
// CHECK-DISASS: cbhhs w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b11AAAAAA,0x65,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 7465fe25

cbheq w5, w5, lbl
// CHECK-INST: cbheq w5, w5, lbl
// CHECK-DISASS: cbheq w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b11AAAAAA,0xc5,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74c5fe05

cbhne w5, w5, lbl
// CHECK-INST: cbhne w5, w5, lbl
// CHECK-DISASS: cbhne w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b11AAAAAA,0xe5,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74e5fde5


///
// CBB<XX>
///
cbbgt w5, w5, lbl
// CHECK-INST: cbbgt w5, w5, lbl
// CHECK-DISASS: cbbgt w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b10AAAAAA,0x05,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 7405bdc5

cbbge w5, w5, lbl
// CHECK-INST: cbbge w5, w5, lbl
// CHECK-DISASS: cbbge w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b10AAAAAA,0x25,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 7425bda5

cbbhi w5, w5, lbl
// CHECK-INST: cbbhi w5, w5, lbl
// CHECK-DISASS: cbbhi w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b10AAAAAA,0x45,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 7445bd85

cbbhs w5, w5, lbl
// CHECK-INST: cbbhs w5, w5, lbl
// CHECK-DISASS: cbbhs w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b10AAAAAA,0x65,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 7465bd65

cbbeq w5, w5, lbl
// CHECK-INST: cbbeq w5, w5, lbl
// CHECK-DISASS: cbbeq w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b10AAAAAA,0xc5,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74c5bd45

cbbne w5, w5, lbl
// CHECK-INST: cbbne w5, w5, lbl
// CHECK-DISASS: cbbne w5, w5, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b10AAAAAA,0xe5,0x74]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 74e5bd25

//------------------------------------------------------------------------------
// Compare & branch (Immediate)
//------------------------------------------------------------------------------

cbgt w5, #63, lbl
// CHECK-INST: cbgt w5, #63, lbl
// CHECK-DISASS: cbgt w5, #63, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b10AAAAAA,0x1f,0x75]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 751fbd05

cbgt x5, #0, lbl
// CHECK-INST: cbgt x5, #0, lbl
// CHECK-DISASS: cbgt x5, #0, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b00AAAAAA,0x00,0xf5]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f5003ce5

cbhi w5, #31, lbl
// CHECK-INST: cbhi w5, #31, lbl
// CHECK-DISASS: cbhi w5, #31, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b10AAAAAA,0x4f,0x75]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 754fbcc5

cbhi x5, #31, lbl
// CHECK-INST: cbhi x5, #31, lbl
// CHECK-DISASS: cbhi x5, #31, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b10AAAAAA,0x4f,0xf5]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f54fbca5

cblt w5, #63, lbl
// CHECK-INST: cblt w5, #63, lbl
// CHECK-DISASS: cblt w5, #63, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b10AAAAAA,0x3f,0x75]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 753fbc85

cblt x5, #0, lbl
// CHECK-INST: cblt x5, #0, lbl
// CHECK-DISASS: cblt x5, #0, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b00AAAAAA,0x20,0xf5]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f5203c65

cblo w5, #31, lbl
// CHECK-INST: cblo w5, #31, lbl
// CHECK-DISASS: cblo w5, #31, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b10AAAAAA,0x6f,0x75]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 756fbc45

cblo x5, #31, lbl
// CHECK-INST: cblo x5, #31, lbl
// CHECK-DISASS: cblo x5, #31, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b10AAAAAA,0x6f,0xf5]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f56fbc25

cbeq w5, #31, lbl
// CHECK-INST: cbeq w5, #31, lbl
// CHECK-DISASS: cbeq w5, #31, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b10AAAAAA,0xcf,0x75]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 75cfbc05

cbeq x5, #31, lbl
// CHECK-INST: cbeq x5, #31, lbl
// CHECK-DISASS: cbeq x5, #31, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b10AAAAAA,0xcf,0xf5]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f5cfbbe5

cbne w5, #31, lbl
// CHECK-INST: cbne w5, #31, lbl
// CHECK-DISASS: cbne w5, #31, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b10AAAAAA,0xef,0x75]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: 75efbbc5

cbne x5, #31, lbl
// CHECK-INST: cbne x5, #31, lbl
// CHECK-DISASS: cbne x5, #31, 0x4 <lbl>
// CHECK-ENCODING: [0bAAA00101,0b10AAAAAA,0xef,0xf5]
// CHECK-ENCODING: fixup A - offset: 0, value: lbl, kind: fixup_aarch64_pcrel_branch9
// CHECK-ERROR: instruction requires: cmpbr
// CHECK-UNKNOWN: f5efbba5
