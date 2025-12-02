// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-mop4,+sme-f8f16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-mop4,+sme-f8f16 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme-mop4,+sme-f8f16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-mop4,+sme-f8f16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme-mop4 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-mop4,+sme-f8f16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme-mop4,+sme-f8f16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// Single vectors

fmop4a  za0.h, z0.b, z16.b  // 10000000-00100000-00000000-00001000
// CHECK-INST: fmop4a  za0.h, z0.b, z16.b
// CHECK-ENCODING: [0x08,0x00,0x20,0x80]
// CHECK-ERROR: instruction requires: sme-f8f16 sme-mop4
// CHECK-UNKNOWN: 80200008 <unknown>

fmop4a  za1.h, z12.b, z24.b  // 10000000-00101000-00000001-10001001
// CHECK-INST: fmop4a  za1.h, z12.b, z24.b
// CHECK-ENCODING: [0x89,0x01,0x28,0x80]
// CHECK-ERROR: instruction requires: sme-f8f16 sme-mop4
// CHECK-UNKNOWN: 80280189 <unknown>

fmop4a  za1.h, z14.b, z30.b  // 10000000-00101110-00000001-11001001
// CHECK-INST: fmop4a  za1.h, z14.b, z30.b
// CHECK-ENCODING: [0xc9,0x01,0x2e,0x80]
// CHECK-ERROR: instruction requires: sme-f8f16 sme-mop4
// CHECK-UNKNOWN: 802e01c9 <unknown>

// Single and multiple vectors

fmop4a  za0.h, z0.b, {z16.b-z17.b}  // 10000000-00110000-00000000-00001000
// CHECK-INST: fmop4a  za0.h, z0.b, { z16.b, z17.b }
// CHECK-ENCODING: [0x08,0x00,0x30,0x80]
// CHECK-ERROR: instruction requires: sme-f8f16 sme-mop4
// CHECK-UNKNOWN: 80300008 <unknown>

fmop4a  za1.h, z10.b, {z20.b-z21.b}  // 10000000-00110100-00000001-01001001
// CHECK-INST: fmop4a  za1.h, z10.b, { z20.b, z21.b }
// CHECK-ENCODING: [0x49,0x01,0x34,0x80]
// CHECK-ERROR: instruction requires: sme-f8f16 sme-mop4
// CHECK-UNKNOWN: 80340149 <unknown>

fmop4a  za1.h, z14.b, {z30.b-z31.b}  // 10000000-00111110-00000001-11001001
// CHECK-INST: fmop4a  za1.h, z14.b, { z30.b, z31.b }
// CHECK-ENCODING: [0xc9,0x01,0x3e,0x80]
// CHECK-ERROR: instruction requires: sme-f8f16 sme-mop4
// CHECK-UNKNOWN: 803e01c9 <unknown>

// Multiple and single vectors

fmop4a  za0.h, {z0.b-z1.b}, z16.b  // 10000000-00100000-00000010-00001000
// CHECK-INST: fmop4a  za0.h, { z0.b, z1.b }, z16.b
// CHECK-ENCODING: [0x08,0x02,0x20,0x80]
// CHECK-ERROR: instruction requires: sme-f8f16 sme-mop4
// CHECK-UNKNOWN: 80200208 <unknown>

fmop4a  za1.h, {z10.b-z11.b}, z20.b  // 10000000-00100100-00000011-01001001
// CHECK-INST: fmop4a  za1.h, { z10.b, z11.b }, z20.b
// CHECK-ENCODING: [0x49,0x03,0x24,0x80]
// CHECK-ERROR: instruction requires: sme-f8f16 sme-mop4
// CHECK-UNKNOWN: 80240349 <unknown>

fmop4a  za1.h, {z14.b-z15.b}, z30.b  // 10000000-00101110-00000011-11001001
// CHECK-INST: fmop4a  za1.h, { z14.b, z15.b }, z30.b
// CHECK-ENCODING: [0xc9,0x03,0x2e,0x80]
// CHECK-ERROR: instruction requires: sme-f8f16 sme-mop4
// CHECK-UNKNOWN: 802e03c9 <unknown>


// Multiple vectors

fmop4a  za0.h, {z0.b-z1.b}, {z16.b-z17.b}  // 10000000-00110000-00000010-00001000
// CHECK-INST: fmop4a  za0.h, { z0.b, z1.b }, { z16.b, z17.b }
// CHECK-ENCODING: [0x08,0x02,0x30,0x80]
// CHECK-ERROR: instruction requires: sme-f8f16 sme-mop4
// CHECK-UNKNOWN: 80300208 <unknown>

fmop4a  za1.h, {z10.b-z11.b}, {z20.b-z21.b}  // 10000000-00110100-00000011-01001001
// CHECK-INST: fmop4a  za1.h, { z10.b, z11.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x49,0x03,0x34,0x80]
// CHECK-ERROR: instruction requires: sme-f8f16 sme-mop4
// CHECK-UNKNOWN: 80340349 <unknown>

fmop4a  za1.h, {z14.b-z15.b}, {z30.b-z31.b}  // 10000000-00111110-00000011-11001001
// CHECK-INST: fmop4a  za1.h, { z14.b, z15.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xc9,0x03,0x3e,0x80]
// CHECK-ERROR: instruction requires: sme-f8f16 sme-mop4
// CHECK-UNKNOWN: 803e03c9 <unknown>

