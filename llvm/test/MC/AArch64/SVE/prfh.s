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

prfh    #0, p0, [x0]
// CHECK-INST: prfh	pldl1keep, p0, [x0]
// CHECK-ENCODING: [0x00,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02000 <unknown>

prfh	pldl1keep, p0, [x0]
// CHECK-INST: prfh	pldl1keep, p0, [x0]
// CHECK-ENCODING: [0x00,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02000 <unknown>

prfh    #1, p0, [x0]
// CHECK-INST: prfh	pldl1strm, p0, [x0]
// CHECK-ENCODING: [0x01,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02001 <unknown>

prfh	pldl1strm, p0, [x0]
// CHECK-INST: prfh	pldl1strm, p0, [x0]
// CHECK-ENCODING: [0x01,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02001 <unknown>

prfh    #2, p0, [x0]
// CHECK-INST: prfh	pldl2keep, p0, [x0]
// CHECK-ENCODING: [0x02,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02002 <unknown>

prfh	pldl2keep, p0, [x0]
// CHECK-INST: prfh	pldl2keep, p0, [x0]
// CHECK-ENCODING: [0x02,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02002 <unknown>

prfh    #3, p0, [x0]
// CHECK-INST: prfh	pldl2strm, p0, [x0]
// CHECK-ENCODING: [0x03,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02003 <unknown>

prfh	pldl2strm, p0, [x0]
// CHECK-INST: prfh	pldl2strm, p0, [x0]
// CHECK-ENCODING: [0x03,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02003 <unknown>

prfh    #4, p0, [x0]
// CHECK-INST: prfh	pldl3keep, p0, [x0]
// CHECK-ENCODING: [0x04,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02004 <unknown>

prfh	pldl3keep, p0, [x0]
// CHECK-INST: prfh	pldl3keep, p0, [x0]
// CHECK-ENCODING: [0x04,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02004 <unknown>

prfh    #5, p0, [x0]
// CHECK-INST: prfh	pldl3strm, p0, [x0]
// CHECK-ENCODING: [0x05,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02005 <unknown>

prfh	pldl3strm, p0, [x0]
// CHECK-INST: prfh	pldl3strm, p0, [x0]
// CHECK-ENCODING: [0x05,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02005 <unknown>

prfh    #6, p0, [x0]
// CHECK-INST: prfh	#6, p0, [x0]
// CHECK-ENCODING: [0x06,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02006 <unknown>

prfh    #7, p0, [x0]
// CHECK-INST: prfh	#7, p0, [x0]
// CHECK-ENCODING: [0x07,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02007 <unknown>

prfh    #8, p0, [x0]
// CHECK-INST: prfh	pstl1keep, p0, [x0]
// CHECK-ENCODING: [0x08,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02008 <unknown>

prfh	pstl1keep, p0, [x0]
// CHECK-INST: prfh	pstl1keep, p0, [x0]
// CHECK-ENCODING: [0x08,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02008 <unknown>

prfh    #9, p0, [x0]
// CHECK-INST: prfh	pstl1strm, p0, [x0]
// CHECK-ENCODING: [0x09,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02009 <unknown>

prfh	pstl1strm, p0, [x0]
// CHECK-INST: prfh	pstl1strm, p0, [x0]
// CHECK-ENCODING: [0x09,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c02009 <unknown>

prfh    #10, p0, [x0]
// CHECK-INST: prfh	pstl2keep, p0, [x0]
// CHECK-ENCODING: [0x0a,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0200a <unknown>

prfh	pstl2keep, p0, [x0]
// CHECK-INST: prfh	pstl2keep, p0, [x0]
// CHECK-ENCODING: [0x0a,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0200a <unknown>

prfh    #11, p0, [x0]
// CHECK-INST: prfh	pstl2strm, p0, [x0]
// CHECK-ENCODING: [0x0b,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0200b <unknown>

prfh	pstl2strm, p0, [x0]
// CHECK-INST: prfh	pstl2strm, p0, [x0]
// CHECK-ENCODING: [0x0b,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0200b <unknown>

prfh    #12, p0, [x0]
// CHECK-INST: prfh	pstl3keep, p0, [x0]
// CHECK-ENCODING: [0x0c,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0200c <unknown>

prfh	pstl3keep, p0, [x0]
// CHECK-INST: prfh	pstl3keep, p0, [x0]
// CHECK-ENCODING: [0x0c,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0200c <unknown>

prfh    #13, p0, [x0]
// CHECK-INST: prfh	pstl3strm, p0, [x0]
// CHECK-ENCODING: [0x0d,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0200d <unknown>

prfh	pstl3strm, p0, [x0]
// CHECK-INST: prfh	pstl3strm, p0, [x0]
// CHECK-ENCODING: [0x0d,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0200d <unknown>

prfh    #14, p0, [x0]
// CHECK-INST: prfh	#14, p0, [x0]
// CHECK-ENCODING: [0x0e,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0200e <unknown>

prfh    #15, p0, [x0]
// CHECK-INST: prfh	#15, p0, [x0]
// CHECK-ENCODING: [0x0f,0x20,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0200f <unknown>

// --------------------------------------------------------------------------//
// Test addressing modes

prfh    pldl1strm, p0, [x0, #-32, mul vl]
// CHECK-INST: prfh     pldl1strm, p0, [x0, #-32, mul vl]
// CHECK-ENCODING: [0x01,0x20,0xe0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85e02001

prfh    pldl1strm, p0, [x0, #31, mul vl]
// CHECK-INST: prfh     pldl1strm, p0, [x0, #31, mul vl]
// CHECK-ENCODING: [0x01,0x20,0xdf,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85df2001
