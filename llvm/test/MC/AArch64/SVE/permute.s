// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+f64mm < %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+sve-perm256 < %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+sve-perm256 < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=+sve,+sve-perm256 - \
// RUN:   | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve,+sve-perm256 < %s \
// RUN:   | llvm-objdump --no-print-imm-hex -d --mattr=+sve,-sve-perm256,-f64mm - \
// RUN:   | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve,+sve-perm256 < %s \
// RUN:   | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:   | llvm-mc -triple=aarch64 -mattr=+sve,+sve-perm256 -disassemble -show-encoding \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+v9.8a < %s \
// RUN:   | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:   | llvm-mc -triple=aarch64 -mattr=+v9.8a -disassemble -show-encoding \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// --------------------------------------------------------------------------//
// ZIP1, ZIP2

zip1 z0.q, z1.q, z2.q
// CHECK-INST: zip1 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x00,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm or sve-perm256
// CHECK-UNKNOWN: 05a20020 <unknown>

zip2 z0.q, z1.q, z2.q
// CHECK-INST: zip2 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x04,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm or sve-perm256
// CHECK-UNKNOWN: 05a20420 <unknown>

// --------------------------------------------------------------------------//
// UZP1, UZP2

uzp1 z0.q, z1.q, z2.q
// CHECK-INST: uzp1 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x08,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm or sve-perm256
// CHECK-UNKNOWN: 05a20820 <unknown>

uzp2 z0.q, z1.q, z2.q
// CHECK-INST: uzp2 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x0c,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm or sve-perm256
// CHECK-UNKNOWN: 05a20c20 <unknown>

// --------------------------------------------------------------------------//
// TRN1, TRN2

trn1 z0.q, z1.q, z2.q
// CHECK-INST: trn1 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x18,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm or sve-perm256
// CHECK-UNKNOWN: 05a21820 <unknown>

trn2 z0.q, z1.q, z2.q
// CHECK-INST: trn2 z0.q, z1.q, z2.q
// CHECK-ENCODING: [0x20,0x1c,0xa2,0x05]
// CHECK-ERROR: instruction requires: f64mm or sve-perm256
// CHECK-UNKNOWN: 05a21c20 <unknown>
