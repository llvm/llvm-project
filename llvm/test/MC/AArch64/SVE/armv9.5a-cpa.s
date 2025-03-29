// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve -mattr=+cpa < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme -mattr=+cpa < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR-NO-SVE
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+cpa < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR-NO-SVE
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR-NO-CPA
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve -mattr=+cpa < %s \
// RUN:        | llvm-objdump -d --mattr=+sve --mattr=+cpa - \
// RUN:        | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve -mattr=+cpa < %s \
// RUN:        | llvm-objdump -d --mattr=+sve --mattr=-cpa - \
// RUN:        | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve -mattr=+cpa < %s \
// RUN:        | llvm-objdump -d --mattr=-sve --mattr=+cpa - \
// RUN:        | FileCheck %s --check-prefix=CHECK-UNKNOWN

addpt z23.d, z13.d, z8.d
// CHECK-INST: addpt z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0x09,0xe8,0x04]
// CHECK-ERROR: instruction requires: cpa sve
// CHECK-ERROR-NO-SVE: instruction requires: sve
// CHECK-ERROR-NO-CPA: instruction requires: cpa
// CHECK-UNKNOWN: 04e809b7 <unknown>

addpt z23.d, p3/m, z23.d, z13.d
// CHECK-INST: addpt z23.d, p3/m, z23.d, z13.d
// CHECK-ENCODING: [0xb7,0x0d,0xc4,0x04]
// CHECK-ERROR: instruction requires: cpa sve
// CHECK-ERROR-NO-SVE: instruction requires: sve
// CHECK-ERROR-NO-CPA: instruction requires: cpa
// CHECK-UNKNOWN: 04c40db7 <unknown>

subpt z23.d, z13.d, z8.d
// CHECK-INST: subpt z23.d, z13.d, z8.d
// CHECK-ENCODING: [0xb7,0x0d,0xe8,0x04]
// CHECK-ERROR: instruction requires: cpa sve
// CHECK-ERROR-NO-SVE: instruction requires: sve
// CHECK-ERROR-NO-CPA: instruction requires: cpa
// CHECK-UNKNOWN: 04e80db7 <unknown>

subpt z23.d, p3/m, z23.d, z13.d
// CHECK-INST: subpt z23.d, p3/m, z23.d, z13.d
// CHECK-ENCODING: [0xb7,0x0d,0xc5,0x04]
// CHECK-ERROR: instruction requires: cpa sve
// CHECK-ERROR-NO-SVE: instruction requires: sve
// CHECK-ERROR-NO-CPA: instruction requires: cpa
// CHECK-UNKNOWN: 04c50db7 <unknown>

madpt z0.d, z1.d, z31.d
// CHECK-INST: madpt z0.d, z1.d, z31.d
// CHECK-ENCODING: [0xe0,0xdb,0xc1,0x44]
// CHECK-ERROR: instruction requires: cpa sve
// CHECK-ERROR-NO-SVE: instruction requires: sve
// CHECK-ERROR-NO-CPA: instruction requires: cpa
// CHECK-UNKNOWN: 44c1dbe0 <unknown>

mlapt z0.d, z1.d, z31.d
// CHECK-INST: mlapt z0.d, z1.d, z31.d
// CHECK-ENCODING: [0x20,0xd0,0xdf,0x44]
// CHECK-ERROR: instruction requires: cpa sve
// CHECK-ERROR-NO-SVE: instruction requires: sve
// CHECK-ERROR-NO-CPA: instruction requires: cpa
// CHECK-UNKNOWN: 44dfd020 <unknown>
