// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// --------------------------------------------------------------------------//
// Test all possible prefetch operation specifiers

prfw    #0, p0, [x0]
// CHECK-INST: prfw	pldl1keep, p0, [x0]
// CHECK-ENCODING: [0x00,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04000 <unknown>

prfw	pldl1keep, p0, [x0]
// CHECK-INST: prfw	pldl1keep, p0, [x0]
// CHECK-ENCODING: [0x00,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04000 <unknown>

prfw    #1, p0, [x0]
// CHECK-INST: prfw	pldl1strm, p0, [x0]
// CHECK-ENCODING: [0x01,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04001 <unknown>

prfw	pldl1strm, p0, [x0]
// CHECK-INST: prfw	pldl1strm, p0, [x0]
// CHECK-ENCODING: [0x01,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04001 <unknown>

prfw    #2, p0, [x0]
// CHECK-INST: prfw	pldl2keep, p0, [x0]
// CHECK-ENCODING: [0x02,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04002 <unknown>

prfw	pldl2keep, p0, [x0]
// CHECK-INST: prfw	pldl2keep, p0, [x0]
// CHECK-ENCODING: [0x02,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04002 <unknown>

prfw    #3, p0, [x0]
// CHECK-INST: prfw	pldl2strm, p0, [x0]
// CHECK-ENCODING: [0x03,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04003 <unknown>

prfw	pldl2strm, p0, [x0]
// CHECK-INST: prfw	pldl2strm, p0, [x0]
// CHECK-ENCODING: [0x03,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04003 <unknown>

prfw    #4, p0, [x0]
// CHECK-INST: prfw	pldl3keep, p0, [x0]
// CHECK-ENCODING: [0x04,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04004 <unknown>

prfw	pldl3keep, p0, [x0]
// CHECK-INST: prfw	pldl3keep, p0, [x0]
// CHECK-ENCODING: [0x04,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04004 <unknown>

prfw    #5, p0, [x0]
// CHECK-INST: prfw	pldl3strm, p0, [x0]
// CHECK-ENCODING: [0x05,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04005 <unknown>

prfw	pldl3strm, p0, [x0]
// CHECK-INST: prfw	pldl3strm, p0, [x0]
// CHECK-ENCODING: [0x05,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04005 <unknown>

prfw    #6, p0, [x0]
// CHECK-INST: prfw	#6, p0, [x0]
// CHECK-ENCODING: [0x06,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04006 <unknown>

prfw    #7, p0, [x0]
// CHECK-INST: prfw	#7, p0, [x0]
// CHECK-ENCODING: [0x07,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04007 <unknown>

prfw    #8, p0, [x0]
// CHECK-INST: prfw	pstl1keep, p0, [x0]
// CHECK-ENCODING: [0x08,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04008 <unknown>

prfw	pstl1keep, p0, [x0]
// CHECK-INST: prfw	pstl1keep, p0, [x0]
// CHECK-ENCODING: [0x08,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04008 <unknown>

prfw    #9, p0, [x0]
// CHECK-INST: prfw	pstl1strm, p0, [x0]
// CHECK-ENCODING: [0x09,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04009 <unknown>

prfw	pstl1strm, p0, [x0]
// CHECK-INST: prfw	pstl1strm, p0, [x0]
// CHECK-ENCODING: [0x09,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c04009 <unknown>

prfw    #10, p0, [x0]
// CHECK-INST: prfw	pstl2keep, p0, [x0]
// CHECK-ENCODING: [0x0a,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0400a <unknown>

prfw	pstl2keep, p0, [x0]
// CHECK-INST: prfw	pstl2keep, p0, [x0]
// CHECK-ENCODING: [0x0a,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0400a <unknown>

prfw    #11, p0, [x0]
// CHECK-INST: prfw	pstl2strm, p0, [x0]
// CHECK-ENCODING: [0x0b,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0400b <unknown>

prfw	pstl2strm, p0, [x0]
// CHECK-INST: prfw	pstl2strm, p0, [x0]
// CHECK-ENCODING: [0x0b,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0400b <unknown>

prfw    #12, p0, [x0]
// CHECK-INST: prfw	pstl3keep, p0, [x0]
// CHECK-ENCODING: [0x0c,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0400c <unknown>

prfw	pstl3keep, p0, [x0]
// CHECK-INST: prfw	pstl3keep, p0, [x0]
// CHECK-ENCODING: [0x0c,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0400c <unknown>

prfw    #13, p0, [x0]
// CHECK-INST: prfw	pstl3strm, p0, [x0]
// CHECK-ENCODING: [0x0d,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0400d <unknown>

prfw	pstl3strm, p0, [x0]
// CHECK-INST: prfw	pstl3strm, p0, [x0]
// CHECK-ENCODING: [0x0d,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0400d <unknown>

prfw    #14, p0, [x0]
// CHECK-INST: prfw	#14, p0, [x0]
// CHECK-ENCODING: [0x0e,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0400e <unknown>

prfw    #15, p0, [x0]
// CHECK-INST: prfw	#15, p0, [x0]
// CHECK-ENCODING: [0x0f,0x40,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0400f <unknown>

// --------------------------------------------------------------------------//
// Test addressing modes

prfw    pldl1strm, p0, [x0, #-32, mul vl]
// CHECK-INST: prfw     pldl1strm, p0, [x0, #-32, mul vl]
// CHECK-ENCODING: [0x01,0x40,0xe0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85e04001

prfw    pldl1strm, p0, [x0, #31, mul vl]
// CHECK-INST: prfw     pldl1strm, p0, [x0, #31, mul vl]
// CHECK-ENCODING: [0x01,0x40,0xdf,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85df4001
