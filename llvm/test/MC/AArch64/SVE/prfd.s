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

prfd    #0, p0, [x0]
// CHECK-INST: prfd	pldl1keep, p0, [x0]
// CHECK-ENCODING: [0x00,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06000 <unknown>

prfd	pldl1keep, p0, [x0]
// CHECK-INST: prfd	pldl1keep, p0, [x0]
// CHECK-ENCODING: [0x00,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06000 <unknown>

prfd    #1, p0, [x0]
// CHECK-INST: prfd	pldl1strm, p0, [x0]
// CHECK-ENCODING: [0x01,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06001 <unknown>

prfd	pldl1strm, p0, [x0]
// CHECK-INST: prfd	pldl1strm, p0, [x0]
// CHECK-ENCODING: [0x01,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06001 <unknown>

prfd    #2, p0, [x0]
// CHECK-INST: prfd	pldl2keep, p0, [x0]
// CHECK-ENCODING: [0x02,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06002 <unknown>

prfd	pldl2keep, p0, [x0]
// CHECK-INST: prfd	pldl2keep, p0, [x0]
// CHECK-ENCODING: [0x02,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06002 <unknown>

prfd    #3, p0, [x0]
// CHECK-INST: prfd	pldl2strm, p0, [x0]
// CHECK-ENCODING: [0x03,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06003 <unknown>

prfd	pldl2strm, p0, [x0]
// CHECK-INST: prfd	pldl2strm, p0, [x0]
// CHECK-ENCODING: [0x03,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06003 <unknown>

prfd    #4, p0, [x0]
// CHECK-INST: prfd	pldl3keep, p0, [x0]
// CHECK-ENCODING: [0x04,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06004 <unknown>

prfd	pldl3keep, p0, [x0]
// CHECK-INST: prfd	pldl3keep, p0, [x0]
// CHECK-ENCODING: [0x04,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06004 <unknown>

prfd    #5, p0, [x0]
// CHECK-INST: prfd	pldl3strm, p0, [x0]
// CHECK-ENCODING: [0x05,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06005 <unknown>

prfd	pldl3strm, p0, [x0]
// CHECK-INST: prfd	pldl3strm, p0, [x0]
// CHECK-ENCODING: [0x05,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06005 <unknown>

prfd    #6, p0, [x0]
// CHECK-INST: prfd	#6, p0, [x0]
// CHECK-ENCODING: [0x06,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06006 <unknown>

prfd    #7, p0, [x0]
// CHECK-INST: prfd	#7, p0, [x0]
// CHECK-ENCODING: [0x07,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06007 <unknown>

prfd    #8, p0, [x0]
// CHECK-INST: prfd	pstl1keep, p0, [x0]
// CHECK-ENCODING: [0x08,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06008 <unknown>

prfd	pstl1keep, p0, [x0]
// CHECK-INST: prfd	pstl1keep, p0, [x0]
// CHECK-ENCODING: [0x08,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06008 <unknown>

prfd    #9, p0, [x0]
// CHECK-INST: prfd	pstl1strm, p0, [x0]
// CHECK-ENCODING: [0x09,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06009 <unknown>

prfd	pstl1strm, p0, [x0]
// CHECK-INST: prfd	pstl1strm, p0, [x0]
// CHECK-ENCODING: [0x09,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c06009 <unknown>

prfd    #10, p0, [x0]
// CHECK-INST: prfd	pstl2keep, p0, [x0]
// CHECK-ENCODING: [0x0a,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0600a <unknown>

prfd	pstl2keep, p0, [x0]
// CHECK-INST: prfd	pstl2keep, p0, [x0]
// CHECK-ENCODING: [0x0a,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0600a <unknown>

prfd    #11, p0, [x0]
// CHECK-INST: prfd	pstl2strm, p0, [x0]
// CHECK-ENCODING: [0x0b,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0600b <unknown>

prfd	pstl2strm, p0, [x0]
// CHECK-INST: prfd	pstl2strm, p0, [x0]
// CHECK-ENCODING: [0x0b,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0600b <unknown>

prfd    #12, p0, [x0]
// CHECK-INST: prfd	pstl3keep, p0, [x0]
// CHECK-ENCODING: [0x0c,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0600c <unknown>

prfd	pstl3keep, p0, [x0]
// CHECK-INST: prfd	pstl3keep, p0, [x0]
// CHECK-ENCODING: [0x0c,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0600c <unknown>

prfd    #13, p0, [x0]
// CHECK-INST: prfd	pstl3strm, p0, [x0]
// CHECK-ENCODING: [0x0d,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0600d <unknown>

prfd	pstl3strm, p0, [x0]
// CHECK-INST: prfd	pstl3strm, p0, [x0]
// CHECK-ENCODING: [0x0d,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0600d <unknown>

prfd    #14, p0, [x0]
// CHECK-INST: prfd	#14, p0, [x0]
// CHECK-ENCODING: [0x0e,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0600e <unknown>

prfd    #15, p0, [x0]
// CHECK-INST: prfd	#15, p0, [x0]
// CHECK-ENCODING: [0x0f,0x60,0xc0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85c0600f <unknown>

// --------------------------------------------------------------------------//
// Test addressing modes

prfd    pldl1strm, p0, [x0, #-32, mul vl]
// CHECK-INST: prfd     pldl1strm, p0, [x0, #-32, mul vl]
// CHECK-ENCODING: [0x01,0x60,0xe0,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85e06001

prfd    pldl1strm, p0, [x0, #31, mul vl]
// CHECK-INST: prfd     pldl1strm, p0, [x0, #31, mul vl]
// CHECK-ENCODING: [0x01,0x60,0xdf,0x85]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 85df6001
