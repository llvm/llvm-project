// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+gcs < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+gcs < %s \
// RUN:        | llvm-objdump -d --mattr=+gcs --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+gcs < %s \
// RUN:        | llvm-objdump -d --mattr=-gcs --no-print-imm-hex - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+gcs < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+gcs -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


msr GCSCR_EL1, x0
// CHECK-INST: msr GCSCR_EL1, x0
// CHECK-ENCODING: encoding: [0x00,0x25,0x18,0xd5]
// CHECK-UNKNOWN:  d5182500 msr GCSCR_EL1, x0

mrs x1, GCSCR_EL1
// CHECK-INST: mrs x1, GCSCR_EL1
// CHECK-ENCODING: encoding: [0x01,0x25,0x38,0xd5]
// CHECK-UNKNOWN:  d5382501 mrs x1, GCSCR_EL1

msr GCSPR_EL1, x2
// CHECK-INST: msr GCSPR_EL1, x2
// CHECK-ENCODING: encoding: [0x22,0x25,0x18,0xd5]
// CHECK-UNKNOWN:  d5182522 msr GCSPR_EL1, x2

mrs x3, GCSPR_EL1
// CHECK-INST: mrs x3, GCSPR_EL1
// CHECK-ENCODING: encoding: [0x23,0x25,0x38,0xd5]
// CHECK-UNKNOWN:  d5382523 mrs x3, GCSPR_EL1

msr GCSCRE0_EL1, x4
// CHECK-INST: msr GCSCRE0_EL1, x4
// CHECK-ENCODING: encoding: [0x44,0x25,0x18,0xd5]
// CHECK-UNKNOWN:  d5182544 msr GCSCRE0_EL1, x4

mrs x5, GCSCRE0_EL1
// CHECK-INST: mrs x5, GCSCRE0_EL1
// CHECK-ENCODING: encoding: [0x45,0x25,0x38,0xd5]
// CHECK-UNKNOWN:  d5382545 mrs x5, GCSCRE0_EL1

msr GCSPR_EL0, x6
// CHECK-INST: msr GCSPR_EL0, x6
// CHECK-ENCODING: encoding: [0x26,0x25,0x1b,0xd5]
// CHECK-UNKNOWN:  d51b2526 msr GCSPR_EL0, x6

mrs x7, GCSPR_EL0
// CHECK-INST: mrs x7, GCSPR_EL0
// CHECK-ENCODING: encoding: [0x27,0x25,0x3b,0xd5]
// CHECK-UNKNOWN:  d53b2527 mrs x7, GCSPR_EL0

msr GCSCR_EL2, x10
// CHECK-INST: msr GCSCR_EL2, x10
// CHECK-ENCODING: encoding: [0x0a,0x25,0x1c,0xd5]
// CHECK-UNKNOWN:  d51c250a msr GCSCR_EL2, x10

mrs x11, GCSCR_EL2
// CHECK-INST: mrs x11, GCSCR_EL2
// CHECK-ENCODING: encoding: [0x0b,0x25,0x3c,0xd5]
// CHECK-UNKNOWN:  d53c250b mrs x11, GCSCR_EL2

msr GCSPR_EL2, x12
// CHECK-INST: msr GCSPR_EL2, x12
// CHECK-ENCODING: encoding: [0x2c,0x25,0x1c,0xd5]
// CHECK-UNKNOWN:  d51c252c msr GCSPR_EL2, x12

mrs x13, GCSPR_EL2
// CHECK-INST: mrs x13, GCSPR_EL2
// CHECK-ENCODING: encoding: [0x2d,0x25,0x3c,0xd5]
// CHECK-UNKNOWN:  d53c252d mrs x13, GCSPR_EL2

msr GCSCR_EL12, x14
// CHECK-INST: msr GCSCR_EL12, x14
// CHECK-ENCODING: encoding: [0x0e,0x25,0x1d,0xd5]
// CHECK-UNKNOWN:  d51d250e msr GCSCR_EL12, x14

mrs x15, GCSCR_EL12
// CHECK-INST: mrs x15, GCSCR_EL12
// CHECK-ENCODING: encoding: [0x0f,0x25,0x3d,0xd5]
// CHECK-UNKNOWN:  d53d250f mrs x15, GCSCR_EL12

msr GCSPR_EL12, x16
// CHECK-INST: msr GCSPR_EL12, x16
// CHECK-ENCODING: encoding: [0x30,0x25,0x1d,0xd5]
// CHECK-UNKNOWN:  d51d2530 msr GCSPR_EL12, x16

mrs x17, GCSPR_EL12
// CHECK-INST: mrs x17, GCSPR_EL12
// CHECK-ENCODING: encoding: [0x31,0x25,0x3d,0xd5]
// CHECK-UNKNOWN:  d53d2531 mrs x17, GCSPR_EL12

msr GCSCR_EL3, x18
// CHECK-INST: msr GCSCR_EL3, x18
// CHECK-ENCODING: encoding: [0x12,0x25,0x1e,0xd5]
// CHECK-UNKNOWN:  d51e2512 msr GCSCR_EL3, x18

mrs x19, GCSCR_EL3
// CHECK-INST: mrs x19, GCSCR_EL3
// CHECK-ENCODING: encoding: [0x13,0x25,0x3e,0xd5]
// CHECK-UNKNOWN:  d53e2513 mrs x19, GCSCR_EL3

msr GCSPR_EL3, x20
// CHECK-INST: msr GCSPR_EL3, x20
// CHECK-ENCODING: encoding: [0x34,0x25,0x1e,0xd5]
// CHECK-UNKNOWN:  d51e2534 msr GCSPR_EL3, x20

mrs x21, GCSPR_EL3
// CHECK-INST: mrs x21, GCSPR_EL3
// CHECK-ENCODING: encoding: [0x35,0x25,0x3e,0xd5]
// CHECK-UNKNOWN:  d53e2535 mrs x21, GCSPR_EL3

gcsss1 x21
// CHECK-INST: gcsss1 x21
// CHECK-ENCODING: encoding: [0x55,0x77,0x0b,0xd5]
// CHECK-ERROR: error: instruction requires: gcs
// CHECK-UNKNOWN:  d50b7755 sys #3, c7, c7, #2, x21

gcsss2 x22
// CHECK-INST: gcsss2 x22
// CHECK-ENCODING: encoding: [0x76,0x77,0x2b,0xd5]
// CHECK-ERROR: error: instruction requires: gcs
// CHECK-UNKNOWN:  d52b7776 sysl x22, #3, c7, c7, #3

gcspushm x25
// CHECK-INST: gcspushm x25
// CHECK-ENCODING: encoding: [0x19,0x77,0x0b,0xd5]
// CHECK-ERROR: error: instruction requires: gcs
// CHECK-UNKNOWN:  d50b7719 sys #3, c7, c7, #0, x25

gcspopm
// CHECK-INST: gcspopm
// CHECK-ENCODING: encoding: [0x3f,0x77,0x2b,0xd5]
// CHECK-ERROR: error: instruction requires: gcs
// CHECK-UNKNOWN:  d52b773f sysl xzr, #3, c7, c7, #1

gcspopm xzr
// CHECK-INST: gcspopm
// CHECK-ENCODING: encoding: [0x3f,0x77,0x2b,0xd5]
// CHECK-ERROR: error: instruction requires: gcs
// CHECK-UNKNOWN:  d52b773f sysl xzr, #3, c7, c7, #1

gcspopm x25
// CHECK-INST: gcspopm x25
// CHECK-ENCODING: encoding: [0x39,0x77,0x2b,0xd5]
// CHECK-ERROR: error: instruction requires: gcs
// CHECK-UNKNOWN:  d52b7739 sysl x25, #3, c7, c7, #1

gcsstr x26, [x27]
// CHECK-INST: gcsstr x26, [x27]
// CHECK-ENCODING: encoding: [0x7a,0x0f,0x1f,0xd9]
// CHECK-ERROR: error: instruction requires: gcs
// CHECK-UNKNOWN:  d91f0f7a <unknown>

gcsstr x26, [sp]
// CHECK-INST: gcsstr x26, [sp]
// CHECK-ENCODING: encoding: [0xfa,0x0f,0x1f,0xd9]
// CHECK-ERROR: error: instruction requires: gcs
// CHECK-UNKNOWN:  d91f0ffa <unknown>

gcssttr x26, [x27]
// CHECK-INST: gcssttr x26, [x27]
// CHECK-ENCODING: encoding: [0x7a,0x1f,0x1f,0xd9]
// CHECK-ERROR: error: instruction requires: gcs
// CHECK-UNKNOWN:  d91f1f7a <unknown>

gcssttr x26, [sp]
// CHECK-INST: gcssttr x26, [sp]
// CHECK-ENCODING: encoding: [0xfa,0x1f,0x1f,0xd9]
// CHECK-ERROR: error: instruction requires: gcs
// CHECK-UNKNOWN:  d91f1ffa <unknown>

gcspushx
// CHECK-INST: gcspushx
// CHECK-ENCODING: encoding: [0x9f,0x77,0x08,0xd5]
// CHECK-ERROR: error: instruction requires: gcs
// CHECK-UNKNOWN:  d508779f sys #0, c7, c7, #4

gcspopcx
// CHECK-INST: gcspopcx
// CHECK-ENCODING: encoding: [0xbf,0x77,0x08,0xd5]
// CHECK-ERROR: error: instruction requires: gcs
// CHECK-UNKNOWN:  d50877bf sys #0, c7, c7, #5

gcspopx
// CHECK-INST: gcspopx
// CHECK-ENCODING: encoding: [0xdf,0x77,0x08,0xd5]
// CHECK-ERROR: error: instruction requires: gcs
// CHECK-UNKNOWN:  d50877df sys #0, c7, c7, #6

gcsb dsync
// CHECK-INST: gcsb dsync
// CHECK-ENCODING: encoding: [0x7f,0x22,0x03,0xd5]
// CHECK-UNKNOWN:  d503227f hint #19
// CHECK-ERROR: hint #19                              // encoding: [0x7f,0x22,0x03,0xd5]

hint #19
// CHECK-INST: gcsb dsync
// CHECK-ENCODING: encoding: [0x7f,0x22,0x03,0xd5]
// CHECK-UNKNOWN:  d503227f hint #19
// CHECK-ERROR: hint #19                              // encoding: [0x7f,0x22,0x03,0xd5]
