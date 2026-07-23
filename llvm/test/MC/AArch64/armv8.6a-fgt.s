// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+v8.6a < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fgt < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fgt < %s \
// RUN:        | llvm-objdump -d --mattr=+fgt - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+fgt < %s \
// RUN:   | llvm-objdump -d --mattr=-fgt - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+fgt < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+fgt -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST



msr HFGRTR_EL2, x0
// CHECK-INST: msr HFGRTR_EL2, x0
// CHECK-ENCODING: encoding: [0x80,0x11,0x1c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:5: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51c1180      msr S3_4_C1_C1_4, x0

msr HFGWTR_EL2, x5
// CHECK-INST: msr HFGWTR_EL2, x5
// CHECK-ENCODING: encoding: [0xa5,0x11,0x1c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:5: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51c11a5      msr S3_4_C1_C1_5, x5

msr HFGITR_EL2, x10
// CHECK-INST: msr HFGITR_EL2, x10
// CHECK-ENCODING: encoding: [0xca,0x11,0x1c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:5: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51c11ca      msr S3_4_C1_C1_6, x10

msr HDFGRTR_EL2, x15
// CHECK-INST: msr HDFGRTR_EL2, x15
// CHECK-ENCODING: encoding: [0x8f,0x31,0x1c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:5: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51c318f      msr S3_4_C3_C1_4, x15

msr HDFGWTR_EL2, x20
// CHECK-INST: msr HDFGWTR_EL2, x20
// CHECK-ENCODING: encoding: [0xb4,0x31,0x1c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:5: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51c31b4      msr S3_4_C3_C1_5, x20

msr HAFGRTR_EL2, x25
// CHECK-INST: msr HAFGRTR_EL2, x25
// CHECK-ENCODING: encoding: [0xd9,0x31,0x1c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:5: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51c31d9      msr S3_4_C3_C1_6, x25

mrs x30,  HFGRTR_EL2
// CHECK-INST: mrs x30, HFGRTR_EL2
// CHECK-ENCODING: encoding: [0x9e,0x11,0x3c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:11: error: expected readable system register
// CHECK-UNKNOWN:  d53c119e      mrs x30, S3_4_C1_C1_4

mrs x25,  HFGWTR_EL2
// CHECK-INST: mrs x25, HFGWTR_EL2
// CHECK-ENCODING: encoding: [0xb9,0x11,0x3c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:11: error: expected readable system register
// CHECK-UNKNOWN:  d53c11b9      mrs x25, S3_4_C1_C1_5

mrs x20,  HFGITR_EL2
// CHECK-INST: mrs x20, HFGITR_EL2
// CHECK-ENCODING: encoding: [0xd4,0x11,0x3c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:11: error: expected readable system register
// CHECK-UNKNOWN:  d53c11d4      mrs x20, S3_4_C1_C1_6

mrs x15,  HDFGRTR_EL2
// CHECK-INST: mrs x15, HDFGRTR_EL2
// CHECK-ENCODING: encoding: [0x8f,0x31,0x3c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:11: error: expected readable system register
// CHECK-UNKNOWN:  d53c318f      mrs x15, S3_4_C3_C1_4

mrs x10,  HDFGWTR_EL2
// CHECK-INST: mrs x10, HDFGWTR_EL2
// CHECK-ENCODING: encoding: [0xaa,0x31,0x3c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:11: error: expected readable system register
// CHECK-UNKNOWN:  d53c31aa      mrs x10, S3_4_C3_C1_5

mrs x5,   HAFGRTR_EL2
// CHECK-INST: mrs x5, HAFGRTR_EL2
// CHECK-ENCODING: encoding: [0xc5,0x31,0x3c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:11: error: expected readable system register
// CHECK-UNKNOWN:  d53c31c5      mrs x5, S3_4_C3_C1_6

mrs x3, HDFGRTR2_EL2
// CHECK-INST: mrs x3, HDFGRTR2_EL2
// CHECK-ENCODING: encoding: [0x03,0x31,0x3c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:9: error: expected readable system register
// CHECK-UNKNOWN:  d53c3103      mrs x3, S3_4_C3_C1_0

mrs x3, HDFGWTR2_EL2
// CHECK-INST: mrs x3, HDFGWTR2_EL2
// CHECK-ENCODING: encoding: [0x23,0x31,0x3c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:9: error: expected readable system register
// CHECK-UNKNOWN:  d53c3123      mrs x3, S3_4_C3_C1_1

mrs x3, HFGRTR2_EL2
// CHECK-INST: mrs x3, HFGRTR2_EL2
// CHECK-ENCODING: encoding: [0x43,0x31,0x3c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:9: error: expected readable system register
// CHECK-UNKNOWN:  d53c3143      mrs x3, S3_4_C3_C1_2

mrs x3, HFGWTR2_EL2
// CHECK-INST: mrs x3, HFGWTR2_EL2
// CHECK-ENCODING: encoding: [0x63,0x31,0x3c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:9: error: expected readable system register
// CHECK-UNKNOWN:  d53c3163      mrs x3, S3_4_C3_C1_3

mrs x3, HFGITR2_EL2
// CHECK-INST: mrs x3, HFGITR2_EL2
// CHECK-ENCODING: encoding: [0xe3,0x31,0x3c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:9: error: expected readable system register
// CHECK-UNKNOWN:  d53c31e3      mrs x3, S3_4_C3_C1_7

msr HDFGRTR2_EL2, x3
// CHECK-INST: msr HDFGRTR2_EL2, x3
// CHECK-ENCODING: encoding: [0x03,0x31,0x1c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:5: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51c3103      msr S3_4_C3_C1_0, x3

msr HDFGWTR2_EL2, x3
// CHECK-INST: msr HDFGWTR2_EL2, x3
// CHECK-ENCODING: encoding: [0x23,0x31,0x1c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:5: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51c3123      msr S3_4_C3_C1_1, x3

msr HFGRTR2_EL2, x3
// CHECK-INST: msr HFGRTR2_EL2, x3
// CHECK-ENCODING: encoding: [0x43,0x31,0x1c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:5: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51c3143      msr S3_4_C3_C1_2, x3

msr HFGWTR2_EL2, x3
// CHECK-INST: msr HFGWTR2_EL2, x3
// CHECK-ENCODING: encoding: [0x63,0x31,0x1c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:5: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51c3163      msr S3_4_C3_C1_3, x3

msr HFGITR2_EL2, x3
// CHECK-INST: msr HFGITR2_EL2, x3
// CHECK-ENCODING: encoding: [0xe3,0x31,0x1c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:5: error: expected writable system register or pstate
// CHECK-UNKNOWN:  d51c31e3      msr S3_4_C3_C1_7, x3
