// RUN: llvm-mc -triple aarch64 -mattr +gcs -show-encoding %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding %s 2>%t | FileCheck %s --check-prefix=NO-GCS
// RUN: FileCheck --check-prefix=ERROR-NO-GCS %s < %t

msr GCSCR_EL1, x0
mrs x1, GCSCR_EL1
// CHECK: msr     GCSCR_EL1, x0                   // encoding: [0x00,0x25,0x18,0xd5]
// CHECK: mrs     x1, GCSCR_EL1                   // encoding: [0x01,0x25,0x38,0xd5]

msr GCSPR_EL1, x2
mrs x3, GCSPR_EL1
// CHECK: msr     GCSPR_EL1, x2                   // encoding: [0x22,0x25,0x18,0xd5]
// CHECK: mrs     x3, GCSPR_EL1                   // encoding: [0x23,0x25,0x38,0xd5]

msr GCSCRE0_EL1, x4
mrs x5, GCSCRE0_EL1
// CHECK: msr     GCSCRE0_EL1, x4                 // encoding: [0x44,0x25,0x18,0xd5]
// CHECK: mrs     x5, GCSCRE0_EL1                 // encoding: [0x45,0x25,0x38,0xd5]

msr GCSPR_EL0, x6
mrs x7, GCSPR_EL0
// CHECK: msr     GCSPR_EL0, x6                   // encoding: [0x26,0x25,0x1b,0xd5]
// CHECK: mrs     x7, GCSPR_EL0                   // encoding: [0x27,0x25,0x3b,0xd5]

msr GCSCR_EL2, x10
mrs x11, GCSCR_EL2
// CHECK: msr     GCSCR_EL2, x10                  // encoding: [0x0a,0x25,0x1c,0xd5]
// CHECK: mrs     x11, GCSCR_EL2                  // encoding: [0x0b,0x25,0x3c,0xd5]

msr GCSPR_EL2, x12
mrs x13, GCSPR_EL2
// CHECK: msr     GCSPR_EL2, x12                  // encoding: [0x2c,0x25,0x1c,0xd5]
// CHECK: mrs     x13, GCSPR_EL2                  // encoding: [0x2d,0x25,0x3c,0xd5]

msr GCSCR_EL12, x14
mrs x15, GCSCR_EL12
// CHECK: msr     GCSCR_EL12, x14                 // encoding: [0x0e,0x25,0x1d,0xd5]
// CHECK: mrs     x15, GCSCR_EL12                 // encoding: [0x0f,0x25,0x3d,0xd5]

msr GCSPR_EL12, x16
mrs x17, GCSPR_EL12
// CHECK: msr     GCSPR_EL12, x16                 // encoding: [0x30,0x25,0x1d,0xd5]
// CHECK: mrs     x17, GCSPR_EL12                 // encoding: [0x31,0x25,0x3d,0xd5]

msr GCSCR_EL3, x18
mrs x19, GCSCR_EL3
// CHECK: msr     GCSCR_EL3, x18                  // encoding: [0x12,0x25,0x1e,0xd5]
// CHECK: mrs     x19, GCSCR_EL3                  // encoding: [0x13,0x25,0x3e,0xd5]

msr GCSPR_EL3, x20
mrs x21, GCSPR_EL3
// CHECK: msr     GCSPR_EL3, x20                  // encoding: [0x34,0x25,0x1e,0xd5]
// CHECK: mrs     x21, GCSPR_EL3                  // encoding: [0x35,0x25,0x3e,0xd5]

gcsss1 x21
// CHECK: gcsss1  x21                        // encoding: [0x55,0x77,0x0b,0xd5]
// ERROR-NO-GCS: [[@LINE-2]]:1: error: instruction requires: gcs

gcsss2 x22
// CHECK: gcsss2  x22                        // encoding: [0x76,0x77,0x2b,0xd5]
// ERROR-NO-GCS: [[@LINE-2]]:1: error: instruction requires: gcs

gcspushm x25
// CHECK: gcspushm x25                       // encoding: [0x19,0x77,0x0b,0xd5]
// ERROR-NO-GCS: [[@LINE-2]]:1: error: instruction requires: gcs

gcspopm
// CHECK: gcspopm                             // encoding: [0x3f,0x77,0x2b,0xd5]
// ERROR-NO-GCS: [[@LINE-2]]:1: error: instruction requires: gcs

gcspopm xzr
// CHECK: gcspopm                            // encoding: [0x3f,0x77,0x2b,0xd5]
// ERROR-NO-GCS: [[@LINE-2]]:1: error: instruction requires: gcs

gcspopm x25
// CHECK: gcspopm  x25                        // encoding: [0x39,0x77,0x2b,0xd5]
// ERROR-NO-GCS: [[@LINE-2]]:1: error: instruction requires: gcs

gcsb dsync
// CHECK: gcsb    dsync                           // encoding: [0x7f,0x22,0x03,0xd5]
// ERROR-NO-GCS-NOT: [[@LINE-2]]:1: error: instruction requires: gcs
// NO-GCS: hint #19                              // encoding: [0x7f,0x22,0x03,0xd5]

hint #19
// CHECK: gcsb    dsync                           // encoding: [0x7f,0x22,0x03,0xd5]
// ERROR-NO-GCS-NOT: [[@LINE-2]]:1: error: instruction requires: gcs
// NO-GCS: hint #19                              // encoding: [0x7f,0x22,0x03,0xd5]

gcsstr x26, [x27]
// CHECK: gcsstr x26, [x27]                        // encoding: [0x7a,0x0f,0x1f,0xd9]
// ERROR-NO-GCS: [[@LINE-2]]:1: error: instruction requires: gcs

gcsstr x26, [sp]
// CHECK: gcsstr x26, [sp]                         // encoding: [0xfa,0x0f,0x1f,0xd9]
// ERROR-NO-GCS: [[@LINE-2]]:1: error: instruction requires: gcs

gcssttr x26, [x27]
// CHECK: gcssttr x26, [x27]                       // encoding: [0x7a,0x1f,0x1f,0xd9]
// ERROR-NO-GCS: [[@LINE-2]]:1: error: instruction requires: gcs

gcssttr x26, [sp]
// CHECK: gcssttr x26, [sp]                        // encoding: [0xfa,0x1f,0x1f,0xd9]
// ERROR-NO-GCS: [[@LINE-2]]:1: error: instruction requires: gcs

gcspushx
// CHECK: gcspushx                          // encoding: [0x9f,0x77,0x08,0xd5]
// ERROR-NO-GCS: [[@LINE-2]]:1: error: instruction requires: gcs

gcspopcx
// CHECK: gcspopcx                          // encoding: [0xbf,0x77,0x08,0xd5]
// ERROR-NO-GCS: [[@LINE-2]]:1: error: instruction requires: gcs

gcspopx
// CHECK: gcspopx                           // encoding: [0xdf,0x77,0x08,0xd5]
// ERROR-NO-GCS: [[@LINE-2]]:1: error: instruction requires: gcs
