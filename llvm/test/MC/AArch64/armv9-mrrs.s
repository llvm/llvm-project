// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+d128,+the,+el2vmsa,+vh < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -mattr=+the,+el2vmsa,+vh -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+d128,+the,+el2vmsa,+vh < %s \
// RUN:        | llvm-objdump -d --mattr=+d128,+the,+el2vmsa,+vh - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+d128,+the,+el2vmsa,+vh < %s \
// RUN:   | llvm-objdump -d --mattr=-d128,+the,+el2vmsa,+vh - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+d128,+the,+el2vmsa,+vh < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+d128,+the,+el2vmsa,+vh -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


// +the required for RCWSMASK_EL1, RCWMASK_EL1
// +el2vmsa required for TTBR0_EL2 (VSCTLR_EL2), VTTBR_EL2
// +vh required for TTBR1_EL2

mrrs  x0, x1, TTBR0_EL1
// CHECK-INST: mrrs x0, x1, TTBR0_EL1
// CHECK-ENCODING: encoding: [0x00,0x20,0x78,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5782000      <unknown>

mrrs  x0, x1, TTBR1_EL1
// CHECK-INST: mrrs x0, x1, TTBR1_EL1
// CHECK-ENCODING: encoding: [0x20,0x20,0x78,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5782020      <unknown>

mrrs  x0, x1, PAR_EL1
// CHECK-INST: mrrs x0, x1, PAR_EL1
// CHECK-ENCODING: encoding: [0x00,0x74,0x78,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d5787400      <unknown>

mrrs  x0, x1, RCWSMASK_EL1
// CHECK-INST: mrrs x0, x1, RCWSMASK_EL1
// CHECK-ENCODING: encoding: [0x60,0xd0,0x78,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d578d060      <unknown>

mrrs  x0, x1, RCWMASK_EL1
// CHECK-INST: mrrs x0, x1, RCWMASK_EL1
// CHECK-ENCODING: encoding: [0xc0,0xd0,0x78,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d578d0c0      <unknown>

mrrs  x0, x1, TTBR0_EL2
// CHECK-INST: mrrs x0, x1, TTBR0_EL2
// CHECK-ENCODING: encoding: [0x00,0x20,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c2000      <unknown>

mrrs  x0, x1, TTBR1_EL2
// CHECK-INST: mrrs x0, x1, TTBR1_EL2
// CHECK-ENCODING: encoding: [0x20,0x20,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c2020      <unknown>

mrrs  x0, x1, VTTBR_EL2
// CHECK-INST: mrrs x0, x1, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x00,0x21,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c2100      <unknown>

mrrs   x0,  x1, VTTBR_EL2
// CHECK-INST: mrrs x0, x1, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x00,0x21,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c2100      <unknown>

mrrs   x2,  x3, VTTBR_EL2
// CHECK-INST: mrrs x2, x3, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x02,0x21,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c2102      <unknown>

mrrs   x4,  x5, VTTBR_EL2
// CHECK-INST: mrrs x4, x5, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x04,0x21,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c2104      <unknown>

mrrs   x6,  x7, VTTBR_EL2
// CHECK-INST: mrrs x6, x7, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x06,0x21,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c2106      <unknown>

mrrs   x8,  x9, VTTBR_EL2
// CHECK-INST: mrrs x8, x9, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x08,0x21,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c2108      <unknown>

mrrs  x10, x11, VTTBR_EL2
// CHECK-INST: mrrs x10, x11, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x0a,0x21,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c210a      <unknown>

mrrs  x12, x13, VTTBR_EL2
// CHECK-INST: mrrs x12, x13, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x0c,0x21,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c210c      <unknown>

mrrs  x14, x15, VTTBR_EL2
// CHECK-INST: mrrs x14, x15, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x0e,0x21,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c210e      <unknown>

mrrs  x16, x17, VTTBR_EL2
// CHECK-INST: mrrs x16, x17, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x10,0x21,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c2110      <unknown>

mrrs  x18, x19, VTTBR_EL2
// CHECK-INST: mrrs x18, x19, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x12,0x21,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c2112      <unknown>

mrrs  x20, x21, VTTBR_EL2
// CHECK-INST: mrrs x20, x21, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x14,0x21,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c2114      <unknown>

mrrs  x22, x23, VTTBR_EL2
// CHECK-INST: mrrs x22, x23, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x16,0x21,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c2116      <unknown>

mrrs  x24, x25, VTTBR_EL2
// CHECK-INST: mrrs x24, x25, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x18,0x21,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c2118      <unknown>

mrrs  x26, x27, VTTBR_EL2
// CHECK-INST: mrrs x26, x27, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x1a,0x21,0x7c,0xd5]
// CHECK-ERROR: error: instruction requires: d128
// CHECK-UNKNOWN:  d57c211a      <unknown>

msrr  TTBR0_EL1, x0, x1
// CHECK-INST: msrr TTBR0_EL1, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x20,0x58,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d5582000      <unknown>

msrr  TTBR1_EL1, x0, x1
// CHECK-INST: msrr TTBR1_EL1, x0, x1
// CHECK-ENCODING: encoding: [0x20,0x20,0x58,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d5582020      <unknown>

msrr  PAR_EL1, x0, x1
// CHECK-INST: msrr PAR_EL1, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x74,0x58,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d5587400      <unknown>

msrr  RCWSMASK_EL1, x0, x1
// CHECK-INST: msrr RCWSMASK_EL1, x0, x1
// CHECK-ENCODING: encoding: [0x60,0xd0,0x58,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d558d060      <unknown>

msrr  RCWMASK_EL1, x0, x1
// CHECK-INST: msrr RCWMASK_EL1, x0, x1
// CHECK-ENCODING: encoding: [0xc0,0xd0,0x58,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d558d0c0      <unknown>

msrr  TTBR0_EL2, x0, x1
// CHECK-INST: msrr TTBR0_EL2, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x20,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c2000      <unknown>

msrr  TTBR1_EL2, x0, x1
// CHECK-INST: msrr TTBR1_EL2, x0, x1
// CHECK-ENCODING: encoding: [0x20,0x20,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c2020      <unknown>

msrr  VTTBR_EL2, x0, x1
// CHECK-INST: msrr VTTBR_EL2, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x21,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c2100      <unknown>

msrr   VTTBR_EL2, x0, x1
// CHECK-INST: msrr VTTBR_EL2, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x21,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c2100      <unknown>

msrr   VTTBR_EL2, x2, x3
// CHECK-INST: msrr VTTBR_EL2, x2, x3
// CHECK-ENCODING: encoding: [0x02,0x21,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c2102      <unknown>

msrr   VTTBR_EL2, x4, x5
// CHECK-INST: msrr VTTBR_EL2, x4, x5
// CHECK-ENCODING: encoding: [0x04,0x21,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c2104      <unknown>

msrr   VTTBR_EL2, x6, x7
// CHECK-INST: msrr VTTBR_EL2, x6, x7
// CHECK-ENCODING: encoding: [0x06,0x21,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c2106      <unknown>

msrr   VTTBR_EL2, x8, x9
// CHECK-INST: msrr VTTBR_EL2, x8, x9
// CHECK-ENCODING: encoding: [0x08,0x21,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c2108      <unknown>

msrr   VTTBR_EL2, x10, x11
// CHECK-INST: msrr VTTBR_EL2, x10, x11
// CHECK-ENCODING: encoding: [0x0a,0x21,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c210a      <unknown>

msrr   VTTBR_EL2, x12, x13
// CHECK-INST: msrr VTTBR_EL2, x12, x13
// CHECK-ENCODING: encoding: [0x0c,0x21,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c210c      <unknown>

msrr   VTTBR_EL2, x14, x15
// CHECK-INST: msrr VTTBR_EL2, x14, x15
// CHECK-ENCODING: encoding: [0x0e,0x21,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c210e      <unknown>

msrr   VTTBR_EL2, x16, x17
// CHECK-INST: msrr VTTBR_EL2, x16, x17
// CHECK-ENCODING: encoding: [0x10,0x21,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c2110      <unknown>

msrr   VTTBR_EL2, x18, x19
// CHECK-INST: msrr VTTBR_EL2, x18, x19
// CHECK-ENCODING: encoding: [0x12,0x21,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c2112      <unknown>

msrr   VTTBR_EL2, x20, x21
// CHECK-INST: msrr VTTBR_EL2, x20, x21
// CHECK-ENCODING: encoding: [0x14,0x21,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c2114      <unknown>

msrr   VTTBR_EL2, x22, x23
// CHECK-INST: msrr VTTBR_EL2, x22, x23
// CHECK-ENCODING: encoding: [0x16,0x21,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c2116      <unknown>

msrr   VTTBR_EL2, x24, x25
// CHECK-INST: msrr VTTBR_EL2, x24, x25
// CHECK-ENCODING: encoding: [0x18,0x21,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c2118      <unknown>

msrr   VTTBR_EL2, x26, x27
// CHECK-INST: msrr VTTBR_EL2, x26, x27
// CHECK-ENCODING: encoding: [0x1a,0x21,0x5c,0xd5]
// CHECK-ERROR: :[[@LINE-3]]:1: error: instruction requires: d128
// CHECK-UNKNOWN:  d55c211a      <unknown>
