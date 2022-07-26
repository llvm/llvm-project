// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN


// ---------------------------------------------------------------------------//
// Test 64-bit form (x0) and its aliases
// ---------------------------------------------------------------------------//
uqdecb  x0
// CHECK-INST: uqdecb  x0
// CHECK-ENCODING: [0xe0,0xff,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430ffe0 <unknown>

uqdecb  x0, all
// CHECK-INST: uqdecb  x0
// CHECK-ENCODING: [0xe0,0xff,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430ffe0 <unknown>

uqdecb  x0, all, mul #1
// CHECK-INST: uqdecb  x0
// CHECK-ENCODING: [0xe0,0xff,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430ffe0 <unknown>

uqdecb  x0, all, mul #16
// CHECK-INST: uqdecb  x0, all, mul #16
// CHECK-ENCODING: [0xe0,0xff,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 043fffe0 <unknown>


// ---------------------------------------------------------------------------//
// Test 32-bit form (w0) and its aliases
// ---------------------------------------------------------------------------//

uqdecb  w0
// CHECK-INST: uqdecb  w0
// CHECK-ENCODING: [0xe0,0xff,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420ffe0 <unknown>

uqdecb  w0, all
// CHECK-INST: uqdecb  w0
// CHECK-ENCODING: [0xe0,0xff,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420ffe0 <unknown>

uqdecb  w0, all, mul #1
// CHECK-INST: uqdecb  w0
// CHECK-ENCODING: [0xe0,0xff,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420ffe0 <unknown>

uqdecb  w0, all, mul #16
// CHECK-INST: uqdecb  w0, all, mul #16
// CHECK-ENCODING: [0xe0,0xff,0x2f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 042fffe0 <unknown>

uqdecb  w0, pow2
// CHECK-INST: uqdecb  w0, pow2
// CHECK-ENCODING: [0x00,0xfc,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0420fc00 <unknown>

uqdecb  w0, pow2, mul #16
// CHECK-INST: uqdecb  w0, pow2, mul #16
// CHECK-ENCODING: [0x00,0xfc,0x2f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 042ffc00 <unknown>


// ---------------------------------------------------------------------------//
// Test all patterns for 64-bit form
// ---------------------------------------------------------------------------//

uqdecb  x0, pow2
// CHECK-INST: uqdecb  x0, pow2
// CHECK-ENCODING: [0x00,0xfc,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fc00 <unknown>

uqdecb  x0, vl1
// CHECK-INST: uqdecb  x0, vl1
// CHECK-ENCODING: [0x20,0xfc,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fc20 <unknown>

uqdecb  x0, vl2
// CHECK-INST: uqdecb  x0, vl2
// CHECK-ENCODING: [0x40,0xfc,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fc40 <unknown>

uqdecb  x0, vl3
// CHECK-INST: uqdecb  x0, vl3
// CHECK-ENCODING: [0x60,0xfc,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fc60 <unknown>

uqdecb  x0, vl4
// CHECK-INST: uqdecb  x0, vl4
// CHECK-ENCODING: [0x80,0xfc,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fc80 <unknown>

uqdecb  x0, vl5
// CHECK-INST: uqdecb  x0, vl5
// CHECK-ENCODING: [0xa0,0xfc,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fca0 <unknown>

uqdecb  x0, vl6
// CHECK-INST: uqdecb  x0, vl6
// CHECK-ENCODING: [0xc0,0xfc,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fcc0 <unknown>

uqdecb  x0, vl7
// CHECK-INST: uqdecb  x0, vl7
// CHECK-ENCODING: [0xe0,0xfc,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fce0 <unknown>

uqdecb  x0, vl8
// CHECK-INST: uqdecb  x0, vl8
// CHECK-ENCODING: [0x00,0xfd,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fd00 <unknown>

uqdecb  x0, vl16
// CHECK-INST: uqdecb  x0, vl16
// CHECK-ENCODING: [0x20,0xfd,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fd20 <unknown>

uqdecb  x0, vl32
// CHECK-INST: uqdecb  x0, vl32
// CHECK-ENCODING: [0x40,0xfd,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fd40 <unknown>

uqdecb  x0, vl64
// CHECK-INST: uqdecb  x0, vl64
// CHECK-ENCODING: [0x60,0xfd,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fd60 <unknown>

uqdecb  x0, vl128
// CHECK-INST: uqdecb  x0, vl128
// CHECK-ENCODING: [0x80,0xfd,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fd80 <unknown>

uqdecb  x0, vl256
// CHECK-INST: uqdecb  x0, vl256
// CHECK-ENCODING: [0xa0,0xfd,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fda0 <unknown>

uqdecb  x0, #14
// CHECK-INST: uqdecb  x0, #14
// CHECK-ENCODING: [0xc0,0xfd,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fdc0 <unknown>

uqdecb  x0, #15
// CHECK-INST: uqdecb  x0, #15
// CHECK-ENCODING: [0xe0,0xfd,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fde0 <unknown>

uqdecb  x0, #16
// CHECK-INST: uqdecb  x0, #16
// CHECK-ENCODING: [0x00,0xfe,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fe00 <unknown>

uqdecb  x0, #17
// CHECK-INST: uqdecb  x0, #17
// CHECK-ENCODING: [0x20,0xfe,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fe20 <unknown>

uqdecb  x0, #18
// CHECK-INST: uqdecb  x0, #18
// CHECK-ENCODING: [0x40,0xfe,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fe40 <unknown>

uqdecb  x0, #19
// CHECK-INST: uqdecb  x0, #19
// CHECK-ENCODING: [0x60,0xfe,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fe60 <unknown>

uqdecb  x0, #20
// CHECK-INST: uqdecb  x0, #20
// CHECK-ENCODING: [0x80,0xfe,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fe80 <unknown>

uqdecb  x0, #21
// CHECK-INST: uqdecb  x0, #21
// CHECK-ENCODING: [0xa0,0xfe,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fea0 <unknown>

uqdecb  x0, #22
// CHECK-INST: uqdecb  x0, #22
// CHECK-ENCODING: [0xc0,0xfe,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fec0 <unknown>

uqdecb  x0, #23
// CHECK-INST: uqdecb  x0, #23
// CHECK-ENCODING: [0xe0,0xfe,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430fee0 <unknown>

uqdecb  x0, #24
// CHECK-INST: uqdecb  x0, #24
// CHECK-ENCODING: [0x00,0xff,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430ff00 <unknown>

uqdecb  x0, #25
// CHECK-INST: uqdecb  x0, #25
// CHECK-ENCODING: [0x20,0xff,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430ff20 <unknown>

uqdecb  x0, #26
// CHECK-INST: uqdecb  x0, #26
// CHECK-ENCODING: [0x40,0xff,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430ff40 <unknown>

uqdecb  x0, #27
// CHECK-INST: uqdecb  x0, #27
// CHECK-ENCODING: [0x60,0xff,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430ff60 <unknown>

uqdecb  x0, #28
// CHECK-INST: uqdecb  x0, #28
// CHECK-ENCODING: [0x80,0xff,0x30,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 0430ff80 <unknown>
