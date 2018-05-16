// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d -mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// --------------------------------------------------------------------------//
// Test all possible prefetch operation specifiers

prfw    #0, p0, [x0]
// CHECK-INST: prfw	pldl1keep, p0, [x0]
// CHECK-ENCODING: [0x00,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 c0 85 <unknown>

prfw	pldl1keep, p0, [x0]
// CHECK-INST: prfw	pldl1keep, p0, [x0]
// CHECK-ENCODING: [0x00,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 c0 85 <unknown>

prfw    #1, p0, [x0]
// CHECK-INST: prfw	pldl1strm, p0, [x0]
// CHECK-ENCODING: [0x01,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 01 40 c0 85 <unknown>

prfw	pldl1strm, p0, [x0]
// CHECK-INST: prfw	pldl1strm, p0, [x0]
// CHECK-ENCODING: [0x01,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 01 40 c0 85 <unknown>

prfw    #2, p0, [x0]
// CHECK-INST: prfw	pldl2keep, p0, [x0]
// CHECK-ENCODING: [0x02,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 02 40 c0 85 <unknown>

prfw	pldl2keep, p0, [x0]
// CHECK-INST: prfw	pldl2keep, p0, [x0]
// CHECK-ENCODING: [0x02,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 02 40 c0 85 <unknown>

prfw    #3, p0, [x0]
// CHECK-INST: prfw	pldl2strm, p0, [x0]
// CHECK-ENCODING: [0x03,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 03 40 c0 85 <unknown>

prfw	pldl2strm, p0, [x0]
// CHECK-INST: prfw	pldl2strm, p0, [x0]
// CHECK-ENCODING: [0x03,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 03 40 c0 85 <unknown>

prfw    #4, p0, [x0]
// CHECK-INST: prfw	pldl3keep, p0, [x0]
// CHECK-ENCODING: [0x04,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 04 40 c0 85 <unknown>

prfw	pldl3keep, p0, [x0]
// CHECK-INST: prfw	pldl3keep, p0, [x0]
// CHECK-ENCODING: [0x04,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 04 40 c0 85 <unknown>

prfw    #5, p0, [x0]
// CHECK-INST: prfw	pldl3strm, p0, [x0]
// CHECK-ENCODING: [0x05,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 05 40 c0 85 <unknown>

prfw	pldl3strm, p0, [x0]
// CHECK-INST: prfw	pldl3strm, p0, [x0]
// CHECK-ENCODING: [0x05,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 05 40 c0 85 <unknown>

prfw    #6, p0, [x0]
// CHECK-INST: prfw	#6, p0, [x0]
// CHECK-ENCODING: [0x06,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 06 40 c0 85 <unknown>

prfw    #7, p0, [x0]
// CHECK-INST: prfw	#7, p0, [x0]
// CHECK-ENCODING: [0x07,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 07 40 c0 85 <unknown>

prfw    #8, p0, [x0]
// CHECK-INST: prfw	pstl1keep, p0, [x0]
// CHECK-ENCODING: [0x08,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 08 40 c0 85 <unknown>

prfw	pstl1keep, p0, [x0]
// CHECK-INST: prfw	pstl1keep, p0, [x0]
// CHECK-ENCODING: [0x08,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 08 40 c0 85 <unknown>

prfw    #9, p0, [x0]
// CHECK-INST: prfw	pstl1strm, p0, [x0]
// CHECK-ENCODING: [0x09,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 09 40 c0 85 <unknown>

prfw	pstl1strm, p0, [x0]
// CHECK-INST: prfw	pstl1strm, p0, [x0]
// CHECK-ENCODING: [0x09,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 09 40 c0 85 <unknown>

prfw    #10, p0, [x0]
// CHECK-INST: prfw	pstl2keep, p0, [x0]
// CHECK-ENCODING: [0x0a,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0a 40 c0 85 <unknown>

prfw	pstl2keep, p0, [x0]
// CHECK-INST: prfw	pstl2keep, p0, [x0]
// CHECK-ENCODING: [0x0a,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0a 40 c0 85 <unknown>

prfw    #11, p0, [x0]
// CHECK-INST: prfw	pstl2strm, p0, [x0]
// CHECK-ENCODING: [0x0b,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0b 40 c0 85 <unknown>

prfw	pstl2strm, p0, [x0]
// CHECK-INST: prfw	pstl2strm, p0, [x0]
// CHECK-ENCODING: [0x0b,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0b 40 c0 85 <unknown>

prfw    #12, p0, [x0]
// CHECK-INST: prfw	pstl3keep, p0, [x0]
// CHECK-ENCODING: [0x0c,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0c 40 c0 85 <unknown>

prfw	pstl3keep, p0, [x0]
// CHECK-INST: prfw	pstl3keep, p0, [x0]
// CHECK-ENCODING: [0x0c,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0c 40 c0 85 <unknown>

prfw    #13, p0, [x0]
// CHECK-INST: prfw	pstl3strm, p0, [x0]
// CHECK-ENCODING: [0x0d,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0d 40 c0 85 <unknown>

prfw	pstl3strm, p0, [x0]
// CHECK-INST: prfw	pstl3strm, p0, [x0]
// CHECK-ENCODING: [0x0d,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0d 40 c0 85 <unknown>

prfw    #14, p0, [x0]
// CHECK-INST: prfw	#14, p0, [x0]
// CHECK-ENCODING: [0x0e,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0e 40 c0 85 <unknown>

prfw    #15, p0, [x0]
// CHECK-INST: prfw	#15, p0, [x0]
// CHECK-ENCODING: [0x0f,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0f 40 c0 85 <unknown>

// --------------------------------------------------------------------------//
// Test addressing modes

prfw    pldl1strm, p0, [x0, #-32, mul vl]
// CHECK-INST: prfw     pldl1strm, p0, [x0, #-32, mul vl]
// CHECK-ENCODING: [0x01,0x40,0xe0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 01 40 e0 85

prfw    pldl1strm, p0, [x0, #31, mul vl]
// CHECK-INST: prfw     pldl1strm, p0, [x0, #31, mul vl]
// CHECK-ENCODING: [0x01,0x40,0xdf,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 01 40 df 85

prfw    pldl1keep, p0, [x0, z0.s, uxtw #2]
// CHECK-INST: prfw    pldl1keep, p0, [x0, z0.s, uxtw #2]
// CHECK-ENCODING: [0x00,0x40,0x20,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 20 84 <unknown>

prfw    pldl3strm, p5, [x10, z21.s, sxtw #2]
// CHECK-INST: prfw    pldl3strm, p5, [x10, z21.s, sxtw #2]
// CHECK-ENCODING: [0x45,0x55,0x75,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 45 55 75 84 <unknown>

prfw    #7, p3, [x13, z8.d, uxtw #2]
// CHECK-INST: prfw    #7, p3, [x13, z8.d, uxtw #2]
// CHECK-ENCODING: [0xa7,0x4d,0x28,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: a7 4d 28 c4 <unknown>

prfw    pldl1keep, p0, [x0, z0.d, sxtw #2]
// CHECK-INST: prfw    pldl1keep, p0, [x0, z0.d, sxtw #2]
// CHECK-ENCODING: [0x00,0x40,0x60,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 40 60 c4 <unknown>

prfw    pldl3strm, p5, [x10, z21.d, lsl #2]
// CHECK-INST: prfw    pldl3strm, p5, [x10, z21.d, lsl #2]
// CHECK-ENCODING: [0x45,0xd5,0x75,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 45 d5 75 c4 <unknown>

prfw    #15, p7, [z31.s, #0]
// CHECK-INST: prfw    #15, p7, [z31.s]
// CHECK-ENCODING: [0xef,0xff,0x00,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef ff 00 85 <unknown>

prfw    #15, p7, [z31.s, #124]
// CHECK-INST: prfw    #15, p7, [z31.s, #124]
// CHECK-ENCODING: [0xef,0xff,0x1f,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef ff 1f 85 <unknown>

prfw    #15, p7, [z31.d, #0]
// CHECK-INST: prfw    #15, p7, [z31.d]
// CHECK-ENCODING: [0xef,0xff,0x00,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef ff 00 c5 <unknown>

prfw    #15, p7, [z31.d, #124]
// CHECK-INST: prfw    #15, p7, [z31.d, #124]
// CHECK-ENCODING: [0xef,0xff,0x1f,0xc5]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef ff 1f c5 <unknown>
