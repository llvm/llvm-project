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

// --------------------------------------------------------------------------//
// Test all possible prefetch operation specifiers

prfb    #0, p0, [x0]
// CHECK-INST: prfb	pldl1keep, p0, [x0]
// CHECK-ENCODING: [0x00,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00000 <unknown>

prfb	pldl1keep, p0, [x0]
// CHECK-INST: prfb	pldl1keep, p0, [x0]
// CHECK-ENCODING: [0x00,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00000 <unknown>

prfb    #1, p0, [x0]
// CHECK-INST: prfb	pldl1strm, p0, [x0]
// CHECK-ENCODING: [0x01,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00001 <unknown>

prfb	pldl1strm, p0, [x0]
// CHECK-INST: prfb	pldl1strm, p0, [x0]
// CHECK-ENCODING: [0x01,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00001 <unknown>

prfb    #2, p0, [x0]
// CHECK-INST: prfb	pldl2keep, p0, [x0]
// CHECK-ENCODING: [0x02,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00002 <unknown>

prfb	pldl2keep, p0, [x0]
// CHECK-INST: prfb	pldl2keep, p0, [x0]
// CHECK-ENCODING: [0x02,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00002 <unknown>

prfb    #3, p0, [x0]
// CHECK-INST: prfb	pldl2strm, p0, [x0]
// CHECK-ENCODING: [0x03,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00003 <unknown>

prfb	pldl2strm, p0, [x0]
// CHECK-INST: prfb	pldl2strm, p0, [x0]
// CHECK-ENCODING: [0x03,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00003 <unknown>

prfb    #4, p0, [x0]
// CHECK-INST: prfb	pldl3keep, p0, [x0]
// CHECK-ENCODING: [0x04,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00004 <unknown>

prfb	pldl3keep, p0, [x0]
// CHECK-INST: prfb	pldl3keep, p0, [x0]
// CHECK-ENCODING: [0x04,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00004 <unknown>

prfb    #5, p0, [x0]
// CHECK-INST: prfb	pldl3strm, p0, [x0]
// CHECK-ENCODING: [0x05,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00005 <unknown>

prfb	pldl3strm, p0, [x0]
// CHECK-INST: prfb	pldl3strm, p0, [x0]
// CHECK-ENCODING: [0x05,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00005 <unknown>

prfb    #6, p0, [x0]
// CHECK-INST: prfb	#6, p0, [x0]
// CHECK-ENCODING: [0x06,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00006 <unknown>

prfb    #7, p0, [x0]
// CHECK-INST: prfb	#7, p0, [x0]
// CHECK-ENCODING: [0x07,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00007 <unknown>

prfb    #8, p0, [x0]
// CHECK-INST: prfb	pstl1keep, p0, [x0]
// CHECK-ENCODING: [0x08,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00008 <unknown>

prfb	pstl1keep, p0, [x0]
// CHECK-INST: prfb	pstl1keep, p0, [x0]
// CHECK-ENCODING: [0x08,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00008 <unknown>

prfb    #9, p0, [x0]
// CHECK-INST: prfb	pstl1strm, p0, [x0]
// CHECK-ENCODING: [0x09,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00009 <unknown>

prfb	pstl1strm, p0, [x0]
// CHECK-INST: prfb	pstl1strm, p0, [x0]
// CHECK-ENCODING: [0x09,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c00009 <unknown>

prfb    #10, p0, [x0]
// CHECK-INST: prfb	pstl2keep, p0, [x0]
// CHECK-ENCODING: [0x0a,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0000a <unknown>

prfb	pstl2keep, p0, [x0]
// CHECK-INST: prfb	pstl2keep, p0, [x0]
// CHECK-ENCODING: [0x0a,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0000a <unknown>

prfb    #11, p0, [x0]
// CHECK-INST: prfb	pstl2strm, p0, [x0]
// CHECK-ENCODING: [0x0b,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0000b <unknown>

prfb	pstl2strm, p0, [x0]
// CHECK-INST: prfb	pstl2strm, p0, [x0]
// CHECK-ENCODING: [0x0b,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0000b <unknown>

prfb    #12, p0, [x0]
// CHECK-INST: prfb	pstl3keep, p0, [x0]
// CHECK-ENCODING: [0x0c,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0000c <unknown>

prfb	pstl3keep, p0, [x0]
// CHECK-INST: prfb	pstl3keep, p0, [x0]
// CHECK-ENCODING: [0x0c,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0000c <unknown>

prfb    #13, p0, [x0]
// CHECK-INST: prfb	pstl3strm, p0, [x0]
// CHECK-ENCODING: [0x0d,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0000d <unknown>

prfb	pstl3strm, p0, [x0]
// CHECK-INST: prfb	pstl3strm, p0, [x0]
// CHECK-ENCODING: [0x0d,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0000d <unknown>

prfb    #14, p0, [x0]
// CHECK-INST: prfb	#14, p0, [x0]
// CHECK-ENCODING: [0x0e,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0000e <unknown>

prfb    #15, p0, [x0]
// CHECK-INST: prfb	#15, p0, [x0]
// CHECK-ENCODING: [0x0f,0x00,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0000f <unknown>

// --------------------------------------------------------------------------//
// Test addressing modes

prfb    #1, p0, [x0, #-32, mul vl]
// CHECK-INST: prfb pldl1strm, p0, [x0, #-32, mul vl]
// CHECK-ENCODING: [0x01,0x00,0xe0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85e00001 <unknown>

prfb    #1, p0, [x0, #31, mul vl]
// CHECK-INST: prfb pldl1strm, p0, [x0, #31, mul vl]
// CHECK-ENCODING: [0x01,0x00,0xdf,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85df0001 <unknown>
