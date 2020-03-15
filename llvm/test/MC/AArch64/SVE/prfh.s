// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// --------------------------------------------------------------------------//
// Test all possible prefetch operation specifiers

prfh    #0, p0, [x0]
// CHECK-INST: prfh	pldl1keep, p0, [x0]
// CHECK-ENCODING: [0x00,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 20 c0 85 <unknown>

prfh	pldl1keep, p0, [x0]
// CHECK-INST: prfh	pldl1keep, p0, [x0]
// CHECK-ENCODING: [0x00,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 20 c0 85 <unknown>

prfh    #1, p0, [x0]
// CHECK-INST: prfh	pldl1strm, p0, [x0]
// CHECK-ENCODING: [0x01,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 01 20 c0 85 <unknown>

prfh	pldl1strm, p0, [x0]
// CHECK-INST: prfh	pldl1strm, p0, [x0]
// CHECK-ENCODING: [0x01,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 01 20 c0 85 <unknown>

prfh    #2, p0, [x0]
// CHECK-INST: prfh	pldl2keep, p0, [x0]
// CHECK-ENCODING: [0x02,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 02 20 c0 85 <unknown>

prfh	pldl2keep, p0, [x0]
// CHECK-INST: prfh	pldl2keep, p0, [x0]
// CHECK-ENCODING: [0x02,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 02 20 c0 85 <unknown>

prfh    #3, p0, [x0]
// CHECK-INST: prfh	pldl2strm, p0, [x0]
// CHECK-ENCODING: [0x03,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 03 20 c0 85 <unknown>

prfh	pldl2strm, p0, [x0]
// CHECK-INST: prfh	pldl2strm, p0, [x0]
// CHECK-ENCODING: [0x03,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 03 20 c0 85 <unknown>

prfh    #4, p0, [x0]
// CHECK-INST: prfh	pldl3keep, p0, [x0]
// CHECK-ENCODING: [0x04,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 04 20 c0 85 <unknown>

prfh	pldl3keep, p0, [x0]
// CHECK-INST: prfh	pldl3keep, p0, [x0]
// CHECK-ENCODING: [0x04,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 04 20 c0 85 <unknown>

prfh    #5, p0, [x0]
// CHECK-INST: prfh	pldl3strm, p0, [x0]
// CHECK-ENCODING: [0x05,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 05 20 c0 85 <unknown>

prfh	pldl3strm, p0, [x0]
// CHECK-INST: prfh	pldl3strm, p0, [x0]
// CHECK-ENCODING: [0x05,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 05 20 c0 85 <unknown>

prfh    #6, p0, [x0]
// CHECK-INST: prfh	#6, p0, [x0]
// CHECK-ENCODING: [0x06,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 06 20 c0 85 <unknown>

prfh    #7, p0, [x0]
// CHECK-INST: prfh	#7, p0, [x0]
// CHECK-ENCODING: [0x07,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 07 20 c0 85 <unknown>

prfh    #8, p0, [x0]
// CHECK-INST: prfh	pstl1keep, p0, [x0]
// CHECK-ENCODING: [0x08,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 08 20 c0 85 <unknown>

prfh	pstl1keep, p0, [x0]
// CHECK-INST: prfh	pstl1keep, p0, [x0]
// CHECK-ENCODING: [0x08,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 08 20 c0 85 <unknown>

prfh    #9, p0, [x0]
// CHECK-INST: prfh	pstl1strm, p0, [x0]
// CHECK-ENCODING: [0x09,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 09 20 c0 85 <unknown>

prfh	pstl1strm, p0, [x0]
// CHECK-INST: prfh	pstl1strm, p0, [x0]
// CHECK-ENCODING: [0x09,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 09 20 c0 85 <unknown>

prfh    #10, p0, [x0]
// CHECK-INST: prfh	pstl2keep, p0, [x0]
// CHECK-ENCODING: [0x0a,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0a 20 c0 85 <unknown>

prfh	pstl2keep, p0, [x0]
// CHECK-INST: prfh	pstl2keep, p0, [x0]
// CHECK-ENCODING: [0x0a,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0a 20 c0 85 <unknown>

prfh    #11, p0, [x0]
// CHECK-INST: prfh	pstl2strm, p0, [x0]
// CHECK-ENCODING: [0x0b,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0b 20 c0 85 <unknown>

prfh	pstl2strm, p0, [x0]
// CHECK-INST: prfh	pstl2strm, p0, [x0]
// CHECK-ENCODING: [0x0b,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0b 20 c0 85 <unknown>

prfh    #12, p0, [x0]
// CHECK-INST: prfh	pstl3keep, p0, [x0]
// CHECK-ENCODING: [0x0c,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0c 20 c0 85 <unknown>

prfh	pstl3keep, p0, [x0]
// CHECK-INST: prfh	pstl3keep, p0, [x0]
// CHECK-ENCODING: [0x0c,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0c 20 c0 85 <unknown>

prfh    #13, p0, [x0]
// CHECK-INST: prfh	pstl3strm, p0, [x0]
// CHECK-ENCODING: [0x0d,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0d 20 c0 85 <unknown>

prfh	pstl3strm, p0, [x0]
// CHECK-INST: prfh	pstl3strm, p0, [x0]
// CHECK-ENCODING: [0x0d,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0d 20 c0 85 <unknown>

prfh    #14, p0, [x0]
// CHECK-INST: prfh	#14, p0, [x0]
// CHECK-ENCODING: [0x0e,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0e 20 c0 85 <unknown>

prfh    #15, p0, [x0]
// CHECK-INST: prfh	#15, p0, [x0]
// CHECK-ENCODING: [0x0f,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 0f 20 c0 85 <unknown>

// --------------------------------------------------------------------------//
// Test addressing modes

prfh    pldl1strm, p0, [x0, #-32, mul vl]
// CHECK-INST: prfh     pldl1strm, p0, [x0, #-32, mul vl]
// CHECK-ENCODING: [0x01,0x20,0xe0,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 01 20 e0 85

prfh    pldl1strm, p0, [x0, #31, mul vl]
// CHECK-INST: prfh     pldl1strm, p0, [x0, #31, mul vl]
// CHECK-ENCODING: [0x01,0x20,0xdf,0x85]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 01 20 df 85

prfh    pldl3strm, p5, [x10, z21.s, uxtw #1]
// CHECK-INST: prfh    pldl3strm, p5, [x10, z21.s, uxtw #1]
// CHECK-ENCODING: [0x45,0x35,0x35,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 45 35 35 84 <unknown>

prfh    pldl3strm, p5, [x10, z21.s, sxtw #1]
// CHECK-INST: prfh    pldl3strm, p5, [x10, z21.s, sxtw #1]
// CHECK-ENCODING: [0x45,0x35,0x75,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 45 35 75 84 <unknown>

prfh    pldl3strm, p5, [x10, z21.d, uxtw #1]
// CHECK-INST: prfh    pldl3strm, p5, [x10, z21.d, uxtw #1]
// CHECK-ENCODING: [0x45,0x35,0x35,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 45 35 35 c4 <unknown>

prfh    pldl3strm, p5, [x10, z21.d, sxtw #1]
// CHECK-INST: prfh    pldl3strm, p5, [x10, z21.d, sxtw #1]
// CHECK-ENCODING: [0x45,0x35,0x75,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 45 35 75 c4 <unknown>

prfh    pldl1keep, p0, [x0, z0.d, lsl #1]
// CHECK-INST: prfh    pldl1keep, p0, [x0, z0.d, lsl #1]
// CHECK-ENCODING: [0x00,0xa0,0x60,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: 00 a0 60 c4 <unknown>

prfh    #15, p7, [z31.s, #0]
// CHECK-INST: prfh    #15, p7, [z31.s]
// CHECK-ENCODING: [0xef,0xff,0x80,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef ff 80 84 <unknown>

prfh    #15, p7, [z31.s, #62]
// CHECK-INST: prfh    #15, p7, [z31.s, #62]
// CHECK-ENCODING: [0xef,0xff,0x9f,0x84]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef ff 9f 84 <unknown>

prfh    #15, p7, [z31.d, #0]
// CHECK-INST: prfh    #15, p7, [z31.d]
// CHECK-ENCODING: [0xef,0xff,0x80,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef ff 80 c4 <unknown>

prfh    #15, p7, [z31.d, #62]
// CHECK-INST: prfh    #15, p7, [z31.d, #62]
// CHECK-ENCODING: [0xef,0xff,0x9f,0xc4]
// CHECK-ERROR: instruction requires: sve
// CHECK-UNKNOWN: ef ff 9f c4 <unknown>
