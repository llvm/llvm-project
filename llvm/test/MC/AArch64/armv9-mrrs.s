// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+the,+el2vmsa,+vh < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+the,+el2vmsa,+vh < %s \
// RUN:        | llvm-objdump -d --mattr=+the,+el2vmsa,+vh - | FileCheck %s --check-prefix=CHECK-INST
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+the,+el2vmsa,+vh < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+the,+el2vmsa,+vh -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


// +the required for RCWSMASK_EL1, RCWMASK_EL1
// +el2vmsa required for TTBR0_EL2 (VSCTLR_EL2), VTTBR_EL2
// +vh required for TTBR1_EL2

mrrs  x0, x1, TTBR0_EL1
// CHECK-INST: mrrs x0, x1, TTBR0_EL1
// CHECK-ENCODING: encoding: [0x00,0x20,0x78,0xd5]

mrrs  x0, x1, TTBR1_EL1
// CHECK-INST: mrrs x0, x1, TTBR1_EL1
// CHECK-ENCODING: encoding: [0x20,0x20,0x78,0xd5]

mrrs  x0, x1, PAR_EL1
// CHECK-INST: mrrs x0, x1, PAR_EL1
// CHECK-ENCODING: encoding: [0x00,0x74,0x78,0xd5]

mrrs  x0, x1, RCWSMASK_EL1
// CHECK-INST: mrrs x0, x1, RCWSMASK_EL1
// CHECK-ENCODING: encoding: [0x60,0xd0,0x78,0xd5]

mrrs  x0, x1, RCWMASK_EL1
// CHECK-INST: mrrs x0, x1, RCWMASK_EL1
// CHECK-ENCODING: encoding: [0xc0,0xd0,0x78,0xd5]

mrrs  x0, x1, TTBR0_EL2
// CHECK-INST: mrrs x0, x1, TTBR0_EL2
// CHECK-ENCODING: encoding: [0x00,0x20,0x7c,0xd5]

mrrs  x0, x1, TTBR1_EL2
// CHECK-INST: mrrs x0, x1, TTBR1_EL2
// CHECK-ENCODING: encoding: [0x20,0x20,0x7c,0xd5]

mrrs  x0, x1, VTTBR_EL2
// CHECK-INST: mrrs x0, x1, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x00,0x21,0x7c,0xd5]

mrrs   x0,  x1, VTTBR_EL2
// CHECK-INST: mrrs x0, x1, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x00,0x21,0x7c,0xd5]

mrrs   x2,  x3, VTTBR_EL2
// CHECK-INST: mrrs x2, x3, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x02,0x21,0x7c,0xd5]

mrrs   x4,  x5, VTTBR_EL2
// CHECK-INST: mrrs x4, x5, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x04,0x21,0x7c,0xd5]

mrrs   x6,  x7, VTTBR_EL2
// CHECK-INST: mrrs x6, x7, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x06,0x21,0x7c,0xd5]

mrrs   x8,  x9, VTTBR_EL2
// CHECK-INST: mrrs x8, x9, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x08,0x21,0x7c,0xd5]

mrrs  x10, x11, VTTBR_EL2
// CHECK-INST: mrrs x10, x11, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x0a,0x21,0x7c,0xd5]

mrrs  x12, x13, VTTBR_EL2
// CHECK-INST: mrrs x12, x13, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x0c,0x21,0x7c,0xd5]

mrrs  x14, x15, VTTBR_EL2
// CHECK-INST: mrrs x14, x15, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x0e,0x21,0x7c,0xd5]

mrrs  x16, x17, VTTBR_EL2
// CHECK-INST: mrrs x16, x17, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x10,0x21,0x7c,0xd5]

mrrs  x18, x19, VTTBR_EL2
// CHECK-INST: mrrs x18, x19, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x12,0x21,0x7c,0xd5]

mrrs  x20, x21, VTTBR_EL2
// CHECK-INST: mrrs x20, x21, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x14,0x21,0x7c,0xd5]

mrrs  x22, x23, VTTBR_EL2
// CHECK-INST: mrrs x22, x23, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x16,0x21,0x7c,0xd5]

mrrs  x24, x25, VTTBR_EL2
// CHECK-INST: mrrs x24, x25, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x18,0x21,0x7c,0xd5]

mrrs  x26, x27, VTTBR_EL2
// CHECK-INST: mrrs x26, x27, VTTBR_EL2
// CHECK-ENCODING: encoding: [0x1a,0x21,0x7c,0xd5]

msrr  TTBR0_EL1, x0, x1
// CHECK-INST: msrr TTBR0_EL1, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x20,0x58,0xd5]

msrr  TTBR1_EL1, x0, x1
// CHECK-INST: msrr TTBR1_EL1, x0, x1
// CHECK-ENCODING: encoding: [0x20,0x20,0x58,0xd5]

msrr  PAR_EL1, x0, x1
// CHECK-INST: msrr PAR_EL1, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x74,0x58,0xd5]

msrr  RCWSMASK_EL1, x0, x1
// CHECK-INST: msrr RCWSMASK_EL1, x0, x1
// CHECK-ENCODING: encoding: [0x60,0xd0,0x58,0xd5]

msrr  RCWMASK_EL1, x0, x1
// CHECK-INST: msrr RCWMASK_EL1, x0, x1
// CHECK-ENCODING: encoding: [0xc0,0xd0,0x58,0xd5]

msrr  TTBR0_EL2, x0, x1
// CHECK-INST: msrr TTBR0_EL2, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x20,0x5c,0xd5]

msrr  TTBR1_EL2, x0, x1
// CHECK-INST: msrr TTBR1_EL2, x0, x1
// CHECK-ENCODING: encoding: [0x20,0x20,0x5c,0xd5]

msrr  VTTBR_EL2, x0, x1
// CHECK-INST: msrr VTTBR_EL2, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x21,0x5c,0xd5]

msrr   VTTBR_EL2, x0, x1
// CHECK-INST: msrr VTTBR_EL2, x0, x1
// CHECK-ENCODING: encoding: [0x00,0x21,0x5c,0xd5]

msrr   VTTBR_EL2, x2, x3
// CHECK-INST: msrr VTTBR_EL2, x2, x3
// CHECK-ENCODING: encoding: [0x02,0x21,0x5c,0xd5]

msrr   VTTBR_EL2, x4, x5
// CHECK-INST: msrr VTTBR_EL2, x4, x5
// CHECK-ENCODING: encoding: [0x04,0x21,0x5c,0xd5]

msrr   VTTBR_EL2, x6, x7
// CHECK-INST: msrr VTTBR_EL2, x6, x7
// CHECK-ENCODING: encoding: [0x06,0x21,0x5c,0xd5]

msrr   VTTBR_EL2, x8, x9
// CHECK-INST: msrr VTTBR_EL2, x8, x9
// CHECK-ENCODING: encoding: [0x08,0x21,0x5c,0xd5]

msrr   VTTBR_EL2, x10, x11
// CHECK-INST: msrr VTTBR_EL2, x10, x11
// CHECK-ENCODING: encoding: [0x0a,0x21,0x5c,0xd5]

msrr   VTTBR_EL2, x12, x13
// CHECK-INST: msrr VTTBR_EL2, x12, x13
// CHECK-ENCODING: encoding: [0x0c,0x21,0x5c,0xd5]

msrr   VTTBR_EL2, x14, x15
// CHECK-INST: msrr VTTBR_EL2, x14, x15
// CHECK-ENCODING: encoding: [0x0e,0x21,0x5c,0xd5]

msrr   VTTBR_EL2, x16, x17
// CHECK-INST: msrr VTTBR_EL2, x16, x17
// CHECK-ENCODING: encoding: [0x10,0x21,0x5c,0xd5]

msrr   VTTBR_EL2, x18, x19
// CHECK-INST: msrr VTTBR_EL2, x18, x19
// CHECK-ENCODING: encoding: [0x12,0x21,0x5c,0xd5]

msrr   VTTBR_EL2, x20, x21
// CHECK-INST: msrr VTTBR_EL2, x20, x21
// CHECK-ENCODING: encoding: [0x14,0x21,0x5c,0xd5]

msrr   VTTBR_EL2, x22, x23
// CHECK-INST: msrr VTTBR_EL2, x22, x23
// CHECK-ENCODING: encoding: [0x16,0x21,0x5c,0xd5]

msrr   VTTBR_EL2, x24, x25
// CHECK-INST: msrr VTTBR_EL2, x24, x25
// CHECK-ENCODING: encoding: [0x18,0x21,0x5c,0xd5]

msrr   VTTBR_EL2, x26, x27
// CHECK-INST: msrr VTTBR_EL2, x26, x27
// CHECK-ENCODING: encoding: [0x1a,0x21,0x5c,0xd5]
