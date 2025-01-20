
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2,+sme-f16f16 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2,+sme-f16f16 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2p2,+sme-f16f16 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p2,+sme-f16f16 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2p2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p2,+sme-f16f16 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p2,+sme-f16f16 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


// FMOP4A

// Single vectors

fmop4a  za0.h, z0.h, z16.h  // 10000001-00000000-00000000-00001000
// CHECK-INST: fmop4a  za0.h, z0.h, z16.h
// CHECK-ENCODING: [0x08,0x00,0x00,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81000008 <unknown>

fmop4a  za1.h, z12.h, z24.h  // 10000001-00001000-00000001-10001001
// CHECK-INST: fmop4a  za1.h, z12.h, z24.h
// CHECK-ENCODING: [0x89,0x01,0x08,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81080189 <unknown>

fmop4a  za1.h, z14.h, z30.h  // 10000001-00001110-00000001-11001001
// CHECK-INST: fmop4a  za1.h, z14.h, z30.h
// CHECK-ENCODING: [0xc9,0x01,0x0e,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 810e01c9 <unknown>

// Single and multiple vectors

fmop4a  za0.h, z0.h, {z16.h-z17.h}  // 10000001-00010000-00000000-00001000
// CHECK-INST: fmop4a  za0.h, z0.h, { z16.h, z17.h }
// CHECK-ENCODING: [0x08,0x00,0x10,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81100008 <unknown>

fmop4a  za1.h, z12.h, {z24.h-z25.h}  // 10000001-00011000-00000001-10001001
// CHECK-INST: fmop4a  za1.h, z12.h, { z24.h, z25.h }
// CHECK-ENCODING: [0x89,0x01,0x18,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81180189 <unknown>

fmop4a  za1.h, z14.h, {z30.h-z31.h}  // 10000001-00011110-00000001-11001001
// CHECK-INST: fmop4a  za1.h, z14.h, { z30.h, z31.h }
// CHECK-ENCODING: [0xc9,0x01,0x1e,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 811e01c9 <unknown>

// Multiple and single vectors

fmop4a  za0.h, {z0.h-z1.h}, z16.h  // 10000001-00000000-00000010-00001000
// CHECK-INST: fmop4a  za0.h, { z0.h, z1.h }, z16.h
// CHECK-ENCODING: [0x08,0x02,0x00,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81000208 <unknown>

fmop4a  za1.h, {z12.h-z13.h}, z24.h  // 10000001-00001000-00000011-10001001
// CHECK-INST: fmop4a  za1.h, { z12.h, z13.h }, z24.h
// CHECK-ENCODING: [0x89,0x03,0x08,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81080389 <unknown>

fmop4a  za1.h, {z14.h-z15.h}, z30.h  // 10000001-00001110-00000011-11001001
// CHECK-INST: fmop4a  za1.h, { z14.h, z15.h }, z30.h
// CHECK-ENCODING: [0xc9,0x03,0x0e,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 810e03c9 <unknown>

// Multiple vectors

fmop4a  za0.h, {z0.h-z1.h}, {z16.h-z17.h}  // 10000001-00010000-00000010-00001000
// CHECK-INST: fmop4a  za0.h, { z0.h, z1.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x08,0x02,0x10,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81100208 <unknown>

fmop4a  za1.h, {z12.h-z13.h}, {z24.h-z25.h}  // 10000001-00011000-00000011-10001001
// CHECK-INST: fmop4a  za1.h, { z12.h, z13.h }, { z24.h, z25.h }
// CHECK-ENCODING: [0x89,0x03,0x18,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81180389 <unknown>

fmop4a  za1.h, {z14.h-z15.h}, {z30.h-z31.h}  // 10000001-00011110-00000011-11001001
// CHECK-INST: fmop4a  za1.h, { z14.h, z15.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xc9,0x03,0x1e,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 811e03c9 <unknown>

// FMOP4S

// Single vectors

fmop4s  za0.h, z0.h, z16.h  // 10000001-00000000-00000000-00011000
// CHECK-INST: fmop4s  za0.h, z0.h, z16.h
// CHECK-ENCODING: [0x18,0x00,0x00,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81000018 <unknown>

fmop4s  za1.h, z12.h, z24.h  // 10000001-00001000-00000001-10011001
// CHECK-INST: fmop4s  za1.h, z12.h, z24.h
// CHECK-ENCODING: [0x99,0x01,0x08,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81080199 <unknown>

fmop4s  za1.h, z14.h, z30.h  // 10000001-00001110-00000001-11011001
// CHECK-INST: fmop4s  za1.h, z14.h, z30.h
// CHECK-ENCODING: [0xd9,0x01,0x0e,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 810e01d9 <unknown>

// Single and multiple vectors

fmop4s  za0.h, z0.h, {z16.h-z17.h}  // 10000001-00010000-00000000-00011000
// CHECK-INST: fmop4s  za0.h, z0.h, { z16.h, z17.h }
// CHECK-ENCODING: [0x18,0x00,0x10,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81100018 <unknown>

fmop4s  za1.h, z12.h, {z24.h-z25.h}  // 10000001-00011000-00000001-10011001
// CHECK-INST: fmop4s  za1.h, z12.h, { z24.h, z25.h }
// CHECK-ENCODING: [0x99,0x01,0x18,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81180199 <unknown>

fmop4s  za1.h, z14.h, {z30.h-z31.h}  // 10000001-00011110-00000001-11011001
// CHECK-INST: fmop4s  za1.h, z14.h, { z30.h, z31.h }
// CHECK-ENCODING: [0xd9,0x01,0x1e,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 811e01d9 <unknown>

// Multiple and single vectors

fmop4s  za0.h, {z0.h-z1.h}, z16.h  // 10000001-00000000-00000010-00011000
// CHECK-INST: fmop4s  za0.h, { z0.h, z1.h }, z16.h
// CHECK-ENCODING: [0x18,0x02,0x00,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81000218 <unknown>

fmop4s  za1.h, {z12.h-z13.h}, z24.h  // 10000001-00001000-00000011-10011001
// CHECK-INST: fmop4s  za1.h, { z12.h, z13.h }, z24.h
// CHECK-ENCODING: [0x99,0x03,0x08,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81080399 <unknown>

fmop4s  za1.h, {z14.h-z15.h}, z30.h  // 10000001-00001110-00000011-11011001
// CHECK-INST: fmop4s  za1.h, { z14.h, z15.h }, z30.h
// CHECK-ENCODING: [0xd9,0x03,0x0e,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 810e03d9 <unknown>

// Multiple vectors

fmop4s  za0.h, {z0.h-z1.h}, {z16.h-z17.h}  // 10000001-00010000-00000010-00011000
// CHECK-INST: fmop4s  za0.h, { z0.h, z1.h }, { z16.h, z17.h }
// CHECK-ENCODING: [0x18,0x02,0x10,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81100218 <unknown>

fmop4s  za1.h, {z12.h-z13.h}, {z24.h-z25.h}  // 10000001-00011000-00000011-10011001
// CHECK-INST: fmop4s  za1.h, { z12.h, z13.h }, { z24.h, z25.h }
// CHECK-ENCODING: [0x99,0x03,0x18,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 81180399 <unknown>

fmop4s  za1.h, {z14.h-z15.h}, {z30.h-z31.h}  // 10000001-00011110-00000011-11011001
// CHECK-INST: fmop4s  za1.h, { z14.h, z15.h }, { z30.h, z31.h }
// CHECK-ENCODING: [0xd9,0x03,0x1e,0x81]
// CHECK-ERROR: instruction requires: sme2p2 sme-f16f16
// CHECK-UNKNOWN: 811e03d9 <unknown>
