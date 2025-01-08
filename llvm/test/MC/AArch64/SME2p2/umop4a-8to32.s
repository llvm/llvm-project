// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

umop4a  za0.s, z0.b, z16.b  // 10000001-00100000-10000000-00000000
// CHECK-INST: umop4a  za0.s, z0.b, z16.b
// CHECK-ENCODING: [0x00,0x80,0x20,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81208000 <unknown>

umop4a  za1.s, z10.b, z20.b  // 10000001-00100100-10000001-01000001
// CHECK-INST: umop4a  za1.s, z10.b, z20.b
// CHECK-ENCODING: [0x41,0x81,0x24,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81248141 <unknown>

umop4a  za3.s, z14.b, z30.b  // 10000001-00101110-10000001-11000011
// CHECK-INST: umop4a  za3.s, z14.b, z30.b
// CHECK-ENCODING: [0xc3,0x81,0x2e,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 812e81c3 <unknown>

umop4a  za0.s, z0.b, {z16.b-z17.b}  // 10000001-00110000-10000000-00000000
// CHECK-INST: umop4a  za0.s, z0.b, { z16.b, z17.b }
// CHECK-ENCODING: [0x00,0x80,0x30,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81308000 <unknown>

umop4a  za1.s, z10.b, {z20.b-z21.b}  // 10000001-00110100-10000001-01000001
// CHECK-INST: umop4a  za1.s, z10.b, { z20.b, z21.b }
// CHECK-ENCODING: [0x41,0x81,0x34,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81348141 <unknown>

umop4a  za3.s, z14.b, {z30.b-z31.b}  // 10000001-00111110-10000001-11000011
// CHECK-INST: umop4a  za3.s, z14.b, { z30.b, z31.b }
// CHECK-ENCODING: [0xc3,0x81,0x3e,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 813e81c3 <unknown>

umop4a  za0.s, {z0.b-z1.b}, z16.b  // 10000001-00100000-10000010-00000000
// CHECK-INST: umop4a  za0.s, { z0.b, z1.b }, z16.b
// CHECK-ENCODING: [0x00,0x82,0x20,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81208200 <unknown>

umop4a  za1.s, {z10.b-z11.b}, z20.b  // 10000001-00100100-10000011-01000001
// CHECK-INST: umop4a  za1.s, { z10.b, z11.b }, z20.b
// CHECK-ENCODING: [0x41,0x83,0x24,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81248341 <unknown>

umop4a  za3.s, {z14.b-z15.b}, z30.b  // 10000001-00101110-10000011-11000011
// CHECK-INST: umop4a  za3.s, { z14.b, z15.b }, z30.b
// CHECK-ENCODING: [0xc3,0x83,0x2e,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 812e83c3 <unknown>

umop4a  za0.s, {z0.b-z1.b}, {z16.b-z17.b}  // 10000001-00110000-10000010-00000000
// CHECK-INST: umop4a  za0.s, { z0.b, z1.b }, { z16.b, z17.b }
// CHECK-ENCODING: [0x00,0x82,0x30,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81308200 <unknown>

umop4a  za1.s, {z10.b-z11.b}, {z20.b-z21.b}  // 10000001-00110100-10000011-01000001
// CHECK-INST: umop4a  za1.s, { z10.b, z11.b }, { z20.b, z21.b }
// CHECK-ENCODING: [0x41,0x83,0x34,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 81348341 <unknown>

umop4a  za3.s, {z14.b-z15.b}, {z30.b-z31.b}  // 10000001-00111110-10000011-11000011
// CHECK-INST: umop4a  za3.s, { z14.b, z15.b }, { z30.b, z31.b }
// CHECK-ENCODING: [0xc3,0x83,0x3e,0x81]
// CHECK-ERROR: instruction requires: sme2p2
// CHECK-UNKNOWN: 813e83c3 <unknown>
