# RUN: llvm-mc -triple=hexagon -mcpu=hexagonv85 -mattr=+hlx -show-encoding %s | FileCheck %s

# Check the HLX register transfer and zero aliases.

# CHECK: L0.h = L1.h
# CHECK-NEXT: encoding: [0x60,0xf8,0x00,0xe5]
{ l0 = l1 }

# CHECK: L1:0.w = L3:2.w
# CHECK-NEXT: encoding: [0xa1,0xf8,0x00,0xe5]
{ l1:0 = l3:2 }

# CHECK: L2 = xor(L15,L15)
# CHECK-NEXT: encoding: [0xe5,0xf7,0x1e,0xe5]
{ l2 = #0 }

# CHECK: L3:2 = xor(L15:14,L15:14)
# CHECK-NEXT: encoding: [0xa4,0xe3,0x3c,0xef]
{ l3:2 = #0 }
