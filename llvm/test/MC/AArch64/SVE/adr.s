// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

adr     z0.s, [z0.s, z0.s]
// CHECK-INST: adr z0.s, [z0.s, z0.s]
// CHECK-ENCODING: [0x00,0xa0,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 04a0a000 <unknown>

adr     z0.s, [z0.s, z0.s, lsl #0]
// CHECK-INST: adr z0.s, [z0.s, z0.s]
// CHECK-ENCODING: [0x00,0xa0,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 04a0a000 <unknown>

adr     z0.s, [z0.s, z0.s, lsl #1]
// CHECK-INST: adr z0.s, [z0.s, z0.s, lsl #1]
// CHECK-ENCODING: [0x00,0xa4,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 04a0a400 <unknown>

adr     z0.s, [z0.s, z0.s, lsl #2]
// CHECK-INST: adr z0.s, [z0.s, z0.s, lsl #2]
// CHECK-ENCODING: [0x00,0xa8,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 04a0a800 <unknown>

adr     z0.s, [z0.s, z0.s, lsl #3]
// CHECK-INST: adr z0.s, [z0.s, z0.s, lsl #3]
// CHECK-ENCODING: [0x00,0xac,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 04a0ac00 <unknown>

adr     z0.d, [z0.d, z0.d]
// CHECK-INST: adr z0.d, [z0.d, z0.d]
// CHECK-ENCODING: [0x00,0xa0,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 04e0a000 <unknown>

adr     z0.d, [z0.d, z0.d, lsl #0]
// CHECK-INST: adr z0.d, [z0.d, z0.d]
// CHECK-ENCODING: [0x00,0xa0,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 04e0a000 <unknown>

adr     z0.d, [z0.d, z0.d, lsl #1]
// CHECK-INST: adr z0.d, [z0.d, z0.d, lsl #1]
// CHECK-ENCODING: [0x00,0xa4,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 04e0a400 <unknown>

adr     z0.d, [z0.d, z0.d, lsl #2]
// CHECK-INST: adr z0.d, [z0.d, z0.d, lsl #2]
// CHECK-ENCODING: [0x00,0xa8,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 04e0a800 <unknown>

adr     z0.d, [z0.d, z0.d, lsl #3]
// CHECK-INST: adr z0.d, [z0.d, z0.d, lsl #3]
// CHECK-ENCODING: [0x00,0xac,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 04e0ac00 <unknown>

adr     z0.d, [z0.d, z0.d, uxtw]
// CHECK-INST: adr z0.d, [z0.d, z0.d, uxtw]
// CHECK-ENCODING: [0x00,0xa0,0x60,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0460a000 <unknown>

adr     z0.d, [z0.d, z0.d, uxtw #0]
// CHECK-INST: adr z0.d, [z0.d, z0.d, uxtw]
// CHECK-ENCODING: [0x00,0xa0,0x60,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0460a000 <unknown>

adr     z0.d, [z0.d, z0.d, uxtw #1]
// CHECK-INST: adr z0.d, [z0.d, z0.d, uxtw #1]
// CHECK-ENCODING: [0x00,0xa4,0x60,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0460a400 <unknown>

adr     z0.d, [z0.d, z0.d, uxtw #2]
// CHECK-INST: adr z0.d, [z0.d, z0.d, uxtw #2]
// CHECK-ENCODING: [0x00,0xa8,0x60,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0460a800 <unknown>

adr     z0.d, [z0.d, z0.d, uxtw #3]
// CHECK-INST: adr z0.d, [z0.d, z0.d, uxtw #3]
// CHECK-ENCODING: [0x00,0xac,0x60,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0460ac00 <unknown>

adr     z0.d, [z0.d, z0.d, sxtw]
// CHECK-INST: adr z0.d, [z0.d, z0.d, sxtw]
// CHECK-ENCODING: [0x00,0xa0,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0420a000 <unknown>

adr     z0.d, [z0.d, z0.d, sxtw #0]
// CHECK-INST: adr z0.d, [z0.d, z0.d, sxtw]
// CHECK-ENCODING: [0x00,0xa0,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0420a000 <unknown>

adr     z0.d, [z0.d, z0.d, sxtw #1]
// CHECK-INST: adr z0.d, [z0.d, z0.d, sxtw #1]
// CHECK-ENCODING: [0x00,0xa4,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0420a400 <unknown>

adr     z0.d, [z0.d, z0.d, sxtw #2]
// CHECK-INST: adr z0.d, [z0.d, z0.d, sxtw #2]
// CHECK-ENCODING: [0x00,0xa8,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0420a800 <unknown>

adr     z0.d, [z0.d, z0.d, sxtw #3]
// CHECK-INST: adr z0.d, [z0.d, z0.d, sxtw #3]
// CHECK-ENCODING: [0x00,0xac,0x20,0x04]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0420ac00 <unknown>
