// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+cmh,+lscp < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+cmh,+lscp < %s \
// RUN:        | llvm-objdump -d --mattr=+cmh,+lscp --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+cmh,+lscp < %s \
// RUN:        | llvm-objdump -d --mattr=-cmh,-lscp --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+cmh,+lscp < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+cmh,+lscp -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// Armv9.7-A Contention Management Hints (FEAT_CMH).

shuh
// CHECK-INST: shuh
// CHECK-ENCODING: encoding: [0x5f,0x26,0x03,0xd5]
// CHECK-ERROR: error: instruction requires: cmh
// CHECK-UNKNOWN: d503265f hint    #50

shuh ph
// CHECK-INST: shuh  ph
// CHECK-ENCODING: encoding: [0x7f,0x26,0x03,0xd5]
// CHECK-ERROR: error: instruction requires: cmh
// CHECK-UNKNOWN: d503267f hint    #51

stcph
// CHECK-INST: stcph
// CHECK-ENCODING: [0x9f,0x26,0x03,0xd5]
// CHECK-ERROR: error: instruction requires: cmh
// CHECK-UNKNOWN: d503269f hint    #52

ldap x0, x1, [x2]
// CHECK-INST: ldap    x0, x1, [x2]
// CHECK-ENCODING: encoding: [0x40,0x58,0x41,0xd9]
// CHECK-ERROR: error: instruction requires: lscp
// CHECK-UNKNOWN: d9415840 <unknown>

ldap x0, x1, [x2, #0]
// CHECK-INST: ldap    x0, x1, [x2]
// CHECK-ENCODING: encoding: [0x40,0x58,0x41,0xd9]
// CHECK-ERROR: error: instruction requires: lscp
// CHECK-UNKNOWN: d9415840 <unknown>

ldapp x0, x1, [x2]
// CHECK-INST: ldapp   x0, x1, [x2]
// CHECK-ENCODING: encoding: [0x40,0x78,0x41,0xd9]
// CHECK-ERROR: error: instruction requires: lscp
// CHECK-UNKNOWN: d9417840 <unknown>

ldapp x0, x1, [x2, #0]
// CHECK-INST: ldapp   x0, x1, [x2]
// CHECK-ENCODING: encoding: [0x40,0x78,0x41,0xd9]
// CHECK-ERROR: error: instruction requires: lscp
// CHECK-UNKNOWN: d9417840 <unknown>

stlp x0, x1, [x2, #0]
// CHECK-INST: stlp    x0, x1, [x2]
// CHECK-ENCODING: encoding: [0x40,0x58,0x01,0xd9]
// CHECK-ERROR: error: instruction requires: lscp
// CHECK-UNKNOWN: d9015840 <unknown>

stlp x0, x1, [x2]
// CHECK-INST: stlp    x0, x1, [x2]
// CHECK-ENCODING: encoding: [0x40,0x58,0x01,0xd9]
// CHECK-ERROR: error: instruction requires: lscp
// CHECK-UNKNOWN: d9015840 <unknown>

mrs x3, VTLBID0_EL2
// CHECK-INST: mrs	x3, VTLBID0_EL2
// CHECK-ENCODING: encoding: [0x03,0x28,0x3c,0xd5]
// CHECK-UNKNOWN: d53c2803
mrs x3, VTLBID1_EL2
// CHECK-INST: mrs	x3, VTLBID1_EL2
// CHECK-ENCODING: encoding: [0x23,0x28,0x3c,0xd5]
// CHECK-UNKNOWN: d53c2823
mrs x3, VTLBID2_EL2
// CHECK-INST: mrs	x3, VTLBID2_EL2
// CHECK-ENCODING: encoding: [0x43,0x28,0x3c,0xd5]
// CHECK-UNKNOWN: d53c2843
mrs x3, VTLBID3_EL2
// CHECK-INST: mrs	x3, VTLBID3_EL2
// CHECK-ENCODING: encoding: [0x63,0x28,0x3c,0xd5]
// CHECK-UNKNOWN: d53c2863
mrs x3, VTLBIDOS0_EL2
// CHECK-INST: mrs	x3, VTLBIDOS0_EL2
// CHECK-ENCODING: encoding: [0x03,0x29,0x3c,0xd5]
// CHECK-UNKNOWN: d53c2903
mrs x3, VTLBIDOS1_EL2
// CHECK-INST: mrs	x3, VTLBIDOS1_EL2
// CHECK-ENCODING: encoding: [0x23,0x29,0x3c,0xd5]
// CHECK-UNKNOWN: d53c2923
mrs x3, VTLBIDOS2_EL2
// CHECK-INST: mrs	x3, VTLBIDOS2_EL2
// CHECK-ENCODING: encoding: [0x43,0x29,0x3c,0xd5]
// CHECK-UNKNOWN: d53c2943
mrs x3, VTLBIDOS3_EL2
// CHECK-INST: mrs	x3, VTLBIDOS3_EL2
// CHECK-ENCODING: encoding: [0x63,0x29,0x3c,0xd5]
// CHECK-UNKNOWN: d53c2963
mrs x3, TLBIDIDR_EL1
// CHECK-INST: mrs	x3, TLBIDIDR_EL1
// CHECK-ENCODING: encoding: [0xc3,0xa4,0x38,0xd5]
// CHECK-UNKNOWN: d538a4c3

msr VTLBID0_EL2, x3
// CHECK-INST: msr	VTLBID0_EL2, x3
// CHECK-ENCODING: encoding: [0x03,0x28,0x1c,0xd5]
// CHECK-UNKNOWN: d51c2803
msr VTLBID1_EL2, x3
// CHECK-INST: msr	VTLBID1_EL2, x3
// CHECK-ENCODING: encoding: [0x23,0x28,0x1c,0xd5]
// CHECK-UNKNOWN: d51c2823
msr VTLBID2_EL2, x3
// CHECK-INST: msr	VTLBID2_EL2, x3
// CHECK-ENCODING: encoding: [0x43,0x28,0x1c,0xd5]
// CHECK-UNKNOWN: d51c2843
msr VTLBID3_EL2, x3
// CHECK-INST: msr	VTLBID3_EL2, x3
// CHECK-ENCODING: encoding: [0x63,0x28,0x1c,0xd5]
// CHECK-UNKNOWN: d51c2863
msr VTLBIDOS0_EL2, x3
// CHECK-INST: msr	VTLBIDOS0_EL2, x3
// CHECK-ENCODING: encoding: [0x03,0x29,0x1c,0xd5]
// CHECK-UNKNOWN:  d51c2903
msr VTLBIDOS1_EL2, x3
// CHECK-INST: msr	VTLBIDOS1_EL2, x3
// CHECK-ENCODING: encoding: [0x23,0x29,0x1c,0xd5]
// CHECK-UNKNOWN:  d51c2923
msr VTLBIDOS2_EL2, x3
// CHECK-INST: msr	VTLBIDOS2_EL2, x3
// CHECK-ENCODING: encoding: [0x43,0x29,0x1c,0xd5]
// CHECK-UNKNOWN:  d51c2943
msr VTLBIDOS3_EL2, x3
// CHECK-INST: msr	VTLBIDOS3_EL2, x3
// CHECK-ENCODING: encoding: [0x63,0x29,0x1c,0xd5]
// CHECK-UNKNOWN:  d51c2963

