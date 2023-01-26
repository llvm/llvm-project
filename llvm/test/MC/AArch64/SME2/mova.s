// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sme2 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST


mova    {z0.h, z1.h}, za0h.h[w12, 0:1]  // 11000000-01000110-00000000-00000000
// CHECK-INST: mov     { z0.h, z1.h }, za0h.h[w12, 0:1]
// CHECK-ENCODING: [0x00,0x00,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0460000 <unknown>

mova    {z20.h, z21.h}, za0h.h[w14, 4:5]  // 11000000-01000110-01000000-01010100
// CHECK-INST: mov     { z20.h, z21.h }, za0h.h[w14, 4:5]
// CHECK-ENCODING: [0x54,0x40,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0464054 <unknown>

mova    {z22.h, z23.h}, za1h.h[w15, 2:3]  // 11000000-01000110-01100000-10110110
// CHECK-INST: mov     { z22.h, z23.h }, za1h.h[w15, 2:3]
// CHECK-ENCODING: [0xb6,0x60,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04660b6 <unknown>

mova    {z30.h, z31.h}, za1h.h[w15, 6:7]  // 11000000-01000110-01100000-11111110
// CHECK-INST: mov     { z30.h, z31.h }, za1h.h[w15, 6:7]
// CHECK-ENCODING: [0xfe,0x60,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04660fe <unknown>

mova    {z4.h, z5.h}, za0h.h[w12, 2:3]  // 11000000-01000110-00000000-00100100
// CHECK-INST: mov     { z4.h, z5.h }, za0h.h[w12, 2:3]
// CHECK-ENCODING: [0x24,0x00,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0460024 <unknown>

mova    {z0.h, z1.h}, za0h.h[w12, 2:3]  // 11000000-01000110-00000000-00100000
// CHECK-INST: mov     { z0.h, z1.h }, za0h.h[w12, 2:3]
// CHECK-ENCODING: [0x20,0x00,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0460020 <unknown>

mova    {z24.h, z25.h}, za0h.h[w14, 6:7]  // 11000000-01000110-01000000-01111000
// CHECK-INST: mov     { z24.h, z25.h }, za0h.h[w14, 6:7]
// CHECK-ENCODING: [0x78,0x40,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0464078 <unknown>

mova    {z0.h, z1.h}, za1h.h[w12, 0:1]  // 11000000-01000110-00000000-10000000
// CHECK-INST: mov     { z0.h, z1.h }, za1h.h[w12, 0:1]
// CHECK-ENCODING: [0x80,0x00,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0460080 <unknown>

mova    {z16.h, z17.h}, za0h.h[w14, 2:3]  // 11000000-01000110-01000000-00110000
// CHECK-INST: mov     { z16.h, z17.h }, za0h.h[w14, 2:3]
// CHECK-ENCODING: [0x30,0x40,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0464030 <unknown>

mova    {z28.h, z29.h}, za1h.h[w12, 4:5]  // 11000000-01000110-00000000-11011100
// CHECK-INST: mov     { z28.h, z29.h }, za1h.h[w12, 4:5]
// CHECK-ENCODING: [0xdc,0x00,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04600dc <unknown>

mova    {z2.h, z3.h}, za0h.h[w15, 2:3]  // 11000000-01000110-01100000-00100010
// CHECK-INST: mov     { z2.h, z3.h }, za0h.h[w15, 2:3]
// CHECK-ENCODING: [0x22,0x60,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0466022 <unknown>

mova    {z6.h, z7.h}, za1h.h[w13, 0:1]  // 11000000-01000110-00100000-10000110
// CHECK-INST: mov     { z6.h, z7.h }, za1h.h[w13, 0:1]
// CHECK-ENCODING: [0x86,0x20,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0462086 <unknown>

// Aliases

mov     {z0.h, z1.h}, za0h.h[w12, 0:1]  // 11000000-01000110-00000000-00000000
// CHECK-INST: mov     { z0.h, z1.h }, za0h.h[w12, 0:1]
// CHECK-ENCODING: [0x00,0x00,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0460000 <unknown>

mov     {z20.h, z21.h}, za0h.h[w14, 4:5]  // 11000000-01000110-01000000-01010100
// CHECK-INST: mov     { z20.h, z21.h }, za0h.h[w14, 4:5]
// CHECK-ENCODING: [0x54,0x40,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0464054 <unknown>

mov     {z22.h, z23.h}, za1h.h[w15, 2:3]  // 11000000-01000110-01100000-10110110
// CHECK-INST: mov     { z22.h, z23.h }, za1h.h[w15, 2:3]
// CHECK-ENCODING: [0xb6,0x60,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04660b6 <unknown>

mov     {z30.h, z31.h}, za1h.h[w15, 6:7]  // 11000000-01000110-01100000-11111110
// CHECK-INST: mov     { z30.h, z31.h }, za1h.h[w15, 6:7]
// CHECK-ENCODING: [0xfe,0x60,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04660fe <unknown>

mov     {z4.h, z5.h}, za0h.h[w12, 2:3]  // 11000000-01000110-00000000-00100100
// CHECK-INST: mov     { z4.h, z5.h }, za0h.h[w12, 2:3]
// CHECK-ENCODING: [0x24,0x00,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0460024 <unknown>

mov     {z0.h, z1.h}, za0h.h[w12, 2:3]  // 11000000-01000110-00000000-00100000
// CHECK-INST: mov     { z0.h, z1.h }, za0h.h[w12, 2:3]
// CHECK-ENCODING: [0x20,0x00,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0460020 <unknown>

mov     {z24.h, z25.h}, za0h.h[w14, 6:7]  // 11000000-01000110-01000000-01111000
// CHECK-INST: mov     { z24.h, z25.h }, za0h.h[w14, 6:7]
// CHECK-ENCODING: [0x78,0x40,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0464078 <unknown>

mov     {z0.h, z1.h}, za1h.h[w12, 0:1]  // 11000000-01000110-00000000-10000000
// CHECK-INST: mov     { z0.h, z1.h }, za1h.h[w12, 0:1]
// CHECK-ENCODING: [0x80,0x00,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0460080 <unknown>

mov     {z16.h, z17.h}, za0h.h[w14, 2:3]  // 11000000-01000110-01000000-00110000
// CHECK-INST: mov     { z16.h, z17.h }, za0h.h[w14, 2:3]
// CHECK-ENCODING: [0x30,0x40,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0464030 <unknown>

mov     {z28.h, z29.h}, za1h.h[w12, 4:5]  // 11000000-01000110-00000000-11011100
// CHECK-INST: mov     { z28.h, z29.h }, za1h.h[w12, 4:5]
// CHECK-ENCODING: [0xdc,0x00,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04600dc <unknown>

mov     {z2.h, z3.h}, za0h.h[w15, 2:3]  // 11000000-01000110-01100000-00100010
// CHECK-INST: mov     { z2.h, z3.h }, za0h.h[w15, 2:3]
// CHECK-ENCODING: [0x22,0x60,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0466022 <unknown>

mov     {z6.h, z7.h}, za1h.h[w13, 0:1]  // 11000000-01000110-00100000-10000110
// CHECK-INST: mov     { z6.h, z7.h }, za1h.h[w13, 0:1]
// CHECK-ENCODING: [0x86,0x20,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0462086 <unknown>


mova    {z0.h, z1.h}, za0v.h[w12, 0:1]  // 11000000-01000110-10000000-00000000
// CHECK-INST: mov     { z0.h, z1.h }, za0v.h[w12, 0:1]
// CHECK-ENCODING: [0x00,0x80,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0468000 <unknown>

mova    {z20.h, z21.h}, za0v.h[w14, 4:5]  // 11000000-01000110-11000000-01010100
// CHECK-INST: mov     { z20.h, z21.h }, za0v.h[w14, 4:5]
// CHECK-ENCODING: [0x54,0xc0,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046c054 <unknown>

mova    {z22.h, z23.h}, za1v.h[w15, 2:3]  // 11000000-01000110-11100000-10110110
// CHECK-INST: mov     { z22.h, z23.h }, za1v.h[w15, 2:3]
// CHECK-ENCODING: [0xb6,0xe0,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046e0b6 <unknown>

mova    {z30.h, z31.h}, za1v.h[w15, 6:7]  // 11000000-01000110-11100000-11111110
// CHECK-INST: mov     { z30.h, z31.h }, za1v.h[w15, 6:7]
// CHECK-ENCODING: [0xfe,0xe0,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046e0fe <unknown>

mova    {z4.h, z5.h}, za0v.h[w12, 2:3]  // 11000000-01000110-10000000-00100100
// CHECK-INST: mov     { z4.h, z5.h }, za0v.h[w12, 2:3]
// CHECK-ENCODING: [0x24,0x80,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0468024 <unknown>

mova    {z0.h, z1.h}, za0v.h[w12, 2:3]  // 11000000-01000110-10000000-00100000
// CHECK-INST: mov     { z0.h, z1.h }, za0v.h[w12, 2:3]
// CHECK-ENCODING: [0x20,0x80,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0468020 <unknown>

mova    {z24.h, z25.h}, za0v.h[w14, 6:7]  // 11000000-01000110-11000000-01111000
// CHECK-INST: mov     { z24.h, z25.h }, za0v.h[w14, 6:7]
// CHECK-ENCODING: [0x78,0xc0,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046c078 <unknown>

mova    {z0.h, z1.h}, za1v.h[w12, 0:1]  // 11000000-01000110-10000000-10000000
// CHECK-INST: mov     { z0.h, z1.h }, za1v.h[w12, 0:1]
// CHECK-ENCODING: [0x80,0x80,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0468080 <unknown>

mova    {z16.h, z17.h}, za0v.h[w14, 2:3]  // 11000000-01000110-11000000-00110000
// CHECK-INST: mov     { z16.h, z17.h }, za0v.h[w14, 2:3]
// CHECK-ENCODING: [0x30,0xc0,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046c030 <unknown>

mova    {z28.h, z29.h}, za1v.h[w12, 4:5]  // 11000000-01000110-10000000-11011100
// CHECK-INST: mov     { z28.h, z29.h }, za1v.h[w12, 4:5]
// CHECK-ENCODING: [0xdc,0x80,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04680dc <unknown>

mova    {z2.h, z3.h}, za0v.h[w15, 2:3]  // 11000000-01000110-11100000-00100010
// CHECK-INST: mov     { z2.h, z3.h }, za0v.h[w15, 2:3]
// CHECK-ENCODING: [0x22,0xe0,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046e022 <unknown>

mova    {z6.h, z7.h}, za1v.h[w13, 0:1]  // 11000000-01000110-10100000-10000110
// CHECK-INST: mov     { z6.h, z7.h }, za1v.h[w13, 0:1]
// CHECK-ENCODING: [0x86,0xa0,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046a086 <unknown>

// Aliases

mov     {z0.h, z1.h}, za0v.h[w12, 0:1]  // 11000000-01000110-10000000-00000000
// CHECK-INST: mov     { z0.h, z1.h }, za0v.h[w12, 0:1]
// CHECK-ENCODING: [0x00,0x80,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0468000 <unknown>

mov     {z20.h, z21.h}, za0v.h[w14, 4:5]  // 11000000-01000110-11000000-01010100
// CHECK-INST: mov     { z20.h, z21.h }, za0v.h[w14, 4:5]
// CHECK-ENCODING: [0x54,0xc0,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046c054 <unknown>

mov     {z22.h, z23.h}, za1v.h[w15, 2:3]  // 11000000-01000110-11100000-10110110
// CHECK-INST: mov     { z22.h, z23.h }, za1v.h[w15, 2:3]
// CHECK-ENCODING: [0xb6,0xe0,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046e0b6 <unknown>

mov     {z30.h, z31.h}, za1v.h[w15, 6:7]  // 11000000-01000110-11100000-11111110
// CHECK-INST: mov     { z30.h, z31.h }, za1v.h[w15, 6:7]
// CHECK-ENCODING: [0xfe,0xe0,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046e0fe <unknown>

mov     {z4.h, z5.h}, za0v.h[w12, 2:3]  // 11000000-01000110-10000000-00100100
// CHECK-INST: mov     { z4.h, z5.h }, za0v.h[w12, 2:3]
// CHECK-ENCODING: [0x24,0x80,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0468024 <unknown>

mov     {z0.h, z1.h}, za0v.h[w12, 2:3]  // 11000000-01000110-10000000-00100000
// CHECK-INST: mov     { z0.h, z1.h }, za0v.h[w12, 2:3]
// CHECK-ENCODING: [0x20,0x80,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0468020 <unknown>

mov     {z24.h, z25.h}, za0v.h[w14, 6:7]  // 11000000-01000110-11000000-01111000
// CHECK-INST: mov     { z24.h, z25.h }, za0v.h[w14, 6:7]
// CHECK-ENCODING: [0x78,0xc0,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046c078 <unknown>

mov     {z0.h, z1.h}, za1v.h[w12, 0:1]  // 11000000-01000110-10000000-10000000
// CHECK-INST: mov     { z0.h, z1.h }, za1v.h[w12, 0:1]
// CHECK-ENCODING: [0x80,0x80,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0468080 <unknown>

mov     {z16.h, z17.h}, za0v.h[w14, 2:3]  // 11000000-01000110-11000000-00110000
// CHECK-INST: mov     { z16.h, z17.h }, za0v.h[w14, 2:3]
// CHECK-ENCODING: [0x30,0xc0,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046c030 <unknown>

mov     {z28.h, z29.h}, za1v.h[w12, 4:5]  // 11000000-01000110-10000000-11011100
// CHECK-INST: mov     { z28.h, z29.h }, za1v.h[w12, 4:5]
// CHECK-ENCODING: [0xdc,0x80,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04680dc <unknown>

mov     {z2.h, z3.h}, za0v.h[w15, 2:3]  // 11000000-01000110-11100000-00100010
// CHECK-INST: mov     { z2.h, z3.h }, za0v.h[w15, 2:3]
// CHECK-ENCODING: [0x22,0xe0,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046e022 <unknown>

mov     {z6.h, z7.h}, za1v.h[w13, 0:1]  // 11000000-01000110-10100000-10000110
// CHECK-INST: mov     { z6.h, z7.h }, za1v.h[w13, 0:1]
// CHECK-ENCODING: [0x86,0xa0,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046a086 <unknown>


mova    za0h.h[w12, 0:1], {z0.h, z1.h}  // 11000000-01000100-00000000-00000000
// CHECK-INST: mov     za0h.h[w12, 0:1], { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0x00,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440000 <unknown>

mova    za1h.h[w14, 2:3], {z10.h, z11.h}  // 11000000-01000100-01000001-01000101
// CHECK-INST: mov     za1h.h[w14, 2:3], { z10.h, z11.h }
// CHECK-ENCODING: [0x45,0x41,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0444145 <unknown>

mova    za1h.h[w15, 6:7], {z12.h, z13.h}  // 11000000-01000100-01100001-10000111
// CHECK-INST: mov     za1h.h[w15, 6:7], { z12.h, z13.h }
// CHECK-ENCODING: [0x87,0x61,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0446187 <unknown>

mova    za1h.h[w15, 6:7], {z30.h, z31.h}  // 11000000-01000100-01100011-11000111
// CHECK-INST: mov     za1h.h[w15, 6:7], { z30.h, z31.h }
// CHECK-ENCODING: [0xc7,0x63,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04463c7 <unknown>

mova    za1h.h[w12, 2:3], {z16.h, z17.h}  // 11000000-01000100-00000010-00000101
// CHECK-INST: mov     za1h.h[w12, 2:3], { z16.h, z17.h }
// CHECK-ENCODING: [0x05,0x02,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440205 <unknown>

mova    za0h.h[w12, 2:3], {z0.h, z1.h}  // 11000000-01000100-00000000-00000001
// CHECK-INST: mov     za0h.h[w12, 2:3], { z0.h, z1.h }
// CHECK-ENCODING: [0x01,0x00,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440001 <unknown>

mova    za0h.h[w14, 0:1], {z18.h, z19.h}  // 11000000-01000100-01000010-01000000
// CHECK-INST: mov     za0h.h[w14, 0:1], { z18.h, z19.h }
// CHECK-ENCODING: [0x40,0x42,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0444240 <unknown>

mova    za0h.h[w12, 0:1], {z12.h, z13.h}  // 11000000-01000100-00000001-10000000
// CHECK-INST: mov     za0h.h[w12, 0:1], { z12.h, z13.h }
// CHECK-ENCODING: [0x80,0x01,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440180 <unknown>

mova    za0h.h[w14, 2:3], {z0.h, z1.h}  // 11000000-01000100-01000000-00000001
// CHECK-INST: mov     za0h.h[w14, 2:3], { z0.h, z1.h }
// CHECK-ENCODING: [0x01,0x40,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0444001 <unknown>

mova    za1h.h[w12, 2:3], {z22.h, z23.h}  // 11000000-01000100-00000010-11000101
// CHECK-INST: mov     za1h.h[w12, 2:3], { z22.h, z23.h }
// CHECK-ENCODING: [0xc5,0x02,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04402c5 <unknown>

mova    za0h.h[w15, 4:5], {z8.h, z9.h}  // 11000000-01000100-01100001-00000010
// CHECK-INST: mov     za0h.h[w15, 4:5], { z8.h, z9.h }
// CHECK-ENCODING: [0x02,0x61,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0446102 <unknown>

mova    za1h.h[w13, 6:7], {z12.h, z13.h}  // 11000000-01000100-00100001-10000111
// CHECK-INST: mov     za1h.h[w13, 6:7], { z12.h, z13.h }
// CHECK-ENCODING: [0x87,0x21,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0442187 <unknown>

// Aliases

mov     za0h.h[w12, 0:1], {z0.h, z1.h}  // 11000000-01000100-00000000-00000000
// CHECK-INST: mov     za0h.h[w12, 0:1], { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0x00,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440000 <unknown>

mov     za1h.h[w14, 2:3], {z10.h, z11.h}  // 11000000-01000100-01000001-01000101
// CHECK-INST: mov     za1h.h[w14, 2:3], { z10.h, z11.h }
// CHECK-ENCODING: [0x45,0x41,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0444145 <unknown>

mov     za1h.h[w15, 6:7], {z12.h, z13.h}  // 11000000-01000100-01100001-10000111
// CHECK-INST: mov     za1h.h[w15, 6:7], { z12.h, z13.h }
// CHECK-ENCODING: [0x87,0x61,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0446187 <unknown>

mov     za1h.h[w15, 6:7], {z30.h, z31.h}  // 11000000-01000100-01100011-11000111
// CHECK-INST: mov     za1h.h[w15, 6:7], { z30.h, z31.h }
// CHECK-ENCODING: [0xc7,0x63,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04463c7 <unknown>

mov     za1h.h[w12, 2:3], {z16.h, z17.h}  // 11000000-01000100-00000010-00000101
// CHECK-INST: mov     za1h.h[w12, 2:3], { z16.h, z17.h }
// CHECK-ENCODING: [0x05,0x02,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440205 <unknown>

mov     za0h.h[w12, 2:3], {z0.h, z1.h}  // 11000000-01000100-00000000-00000001
// CHECK-INST: mov     za0h.h[w12, 2:3], { z0.h, z1.h }
// CHECK-ENCODING: [0x01,0x00,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440001 <unknown>

mov     za0h.h[w14, 0:1], {z18.h, z19.h}  // 11000000-01000100-01000010-01000000
// CHECK-INST: mov     za0h.h[w14, 0:1], { z18.h, z19.h }
// CHECK-ENCODING: [0x40,0x42,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0444240 <unknown>

mov     za0h.h[w12, 0:1], {z12.h, z13.h}  // 11000000-01000100-00000001-10000000
// CHECK-INST: mov     za0h.h[w12, 0:1], { z12.h, z13.h }
// CHECK-ENCODING: [0x80,0x01,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440180 <unknown>

mov     za0h.h[w14, 2:3], {z0.h, z1.h}  // 11000000-01000100-01000000-00000001
// CHECK-INST: mov     za0h.h[w14, 2:3], { z0.h, z1.h }
// CHECK-ENCODING: [0x01,0x40,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0444001 <unknown>

mov     za1h.h[w12, 2:3], {z22.h, z23.h}  // 11000000-01000100-00000010-11000101
// CHECK-INST: mov     za1h.h[w12, 2:3], { z22.h, z23.h }
// CHECK-ENCODING: [0xc5,0x02,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04402c5 <unknown>

mov     za0h.h[w15, 4:5], {z8.h, z9.h}  // 11000000-01000100-01100001-00000010
// CHECK-INST: mov     za0h.h[w15, 4:5], { z8.h, z9.h }
// CHECK-ENCODING: [0x02,0x61,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0446102 <unknown>

mov     za1h.h[w13, 6:7], {z12.h, z13.h}  // 11000000-01000100-00100001-10000111
// CHECK-INST: mov     za1h.h[w13, 6:7], { z12.h, z13.h }
// CHECK-ENCODING: [0x87,0x21,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0442187 <unknown>


mova    za0v.h[w12, 0:1], {z0.h, z1.h}  // 11000000-01000100-10000000-00000000
// CHECK-INST: mov     za0v.h[w12, 0:1], { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0x80,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448000 <unknown>

mova    za1v.h[w14, 2:3], {z10.h, z11.h}  // 11000000-01000100-11000001-01000101
// CHECK-INST: mov     za1v.h[w14, 2:3], { z10.h, z11.h }
// CHECK-ENCODING: [0x45,0xc1,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044c145 <unknown>

mova    za1v.h[w15, 6:7], {z12.h, z13.h}  // 11000000-01000100-11100001-10000111
// CHECK-INST: mov     za1v.h[w15, 6:7], { z12.h, z13.h }
// CHECK-ENCODING: [0x87,0xe1,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044e187 <unknown>

mova    za1v.h[w15, 6:7], {z30.h, z31.h}  // 11000000-01000100-11100011-11000111
// CHECK-INST: mov     za1v.h[w15, 6:7], { z30.h, z31.h }
// CHECK-ENCODING: [0xc7,0xe3,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044e3c7 <unknown>

mova    za1v.h[w12, 2:3], {z16.h, z17.h}  // 11000000-01000100-10000010-00000101
// CHECK-INST: mov     za1v.h[w12, 2:3], { z16.h, z17.h }
// CHECK-ENCODING: [0x05,0x82,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448205 <unknown>

mova    za0v.h[w12, 2:3], {z0.h, z1.h}  // 11000000-01000100-10000000-00000001
// CHECK-INST: mov     za0v.h[w12, 2:3], { z0.h, z1.h }
// CHECK-ENCODING: [0x01,0x80,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448001 <unknown>

mova    za0v.h[w14, 0:1], {z18.h, z19.h}  // 11000000-01000100-11000010-01000000
// CHECK-INST: mov     za0v.h[w14, 0:1], { z18.h, z19.h }
// CHECK-ENCODING: [0x40,0xc2,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044c240 <unknown>

mova    za0v.h[w12, 0:1], {z12.h, z13.h}  // 11000000-01000100-10000001-10000000
// CHECK-INST: mov     za0v.h[w12, 0:1], { z12.h, z13.h }
// CHECK-ENCODING: [0x80,0x81,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448180 <unknown>

mova    za0v.h[w14, 2:3], {z0.h, z1.h}  // 11000000-01000100-11000000-00000001
// CHECK-INST: mov     za0v.h[w14, 2:3], { z0.h, z1.h }
// CHECK-ENCODING: [0x01,0xc0,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044c001 <unknown>

mova    za1v.h[w12, 2:3], {z22.h, z23.h}  // 11000000-01000100-10000010-11000101
// CHECK-INST: mov     za1v.h[w12, 2:3], { z22.h, z23.h }
// CHECK-ENCODING: [0xc5,0x82,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04482c5 <unknown>

mova    za0v.h[w15, 4:5], {z8.h, z9.h}  // 11000000-01000100-11100001-00000010
// CHECK-INST: mov     za0v.h[w15, 4:5], { z8.h, z9.h }
// CHECK-ENCODING: [0x02,0xe1,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044e102 <unknown>

mova    za1v.h[w13, 6:7], {z12.h, z13.h}  // 11000000-01000100-10100001-10000111
// CHECK-INST: mov     za1v.h[w13, 6:7], { z12.h, z13.h }
// CHECK-ENCODING: [0x87,0xa1,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044a187 <unknown>

// Aliases

mov     za0v.h[w12, 0:1], {z0.h, z1.h}  // 11000000-01000100-10000000-00000000
// CHECK-INST: mov     za0v.h[w12, 0:1], { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0x80,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448000 <unknown>

mov     za1v.h[w14, 2:3], {z10.h, z11.h}  // 11000000-01000100-11000001-01000101
// CHECK-INST: mov     za1v.h[w14, 2:3], { z10.h, z11.h }
// CHECK-ENCODING: [0x45,0xc1,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044c145 <unknown>

mov     za1v.h[w15, 6:7], {z12.h, z13.h}  // 11000000-01000100-11100001-10000111
// CHECK-INST: mov     za1v.h[w15, 6:7], { z12.h, z13.h }
// CHECK-ENCODING: [0x87,0xe1,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044e187 <unknown>

mov     za1v.h[w15, 6:7], {z30.h, z31.h}  // 11000000-01000100-11100011-11000111
// CHECK-INST: mov     za1v.h[w15, 6:7], { z30.h, z31.h }
// CHECK-ENCODING: [0xc7,0xe3,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044e3c7 <unknown>

mov     za1v.h[w12, 2:3], {z16.h, z17.h}  // 11000000-01000100-10000010-00000101
// CHECK-INST: mov     za1v.h[w12, 2:3], { z16.h, z17.h }
// CHECK-ENCODING: [0x05,0x82,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448205 <unknown>

mov     za0v.h[w12, 2:3], {z0.h, z1.h}  // 11000000-01000100-10000000-00000001
// CHECK-INST: mov     za0v.h[w12, 2:3], { z0.h, z1.h }
// CHECK-ENCODING: [0x01,0x80,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448001 <unknown>

mov     za0v.h[w14, 0:1], {z18.h, z19.h}  // 11000000-01000100-11000010-01000000
// CHECK-INST: mov     za0v.h[w14, 0:1], { z18.h, z19.h }
// CHECK-ENCODING: [0x40,0xc2,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044c240 <unknown>

mov     za0v.h[w12, 0:1], {z12.h, z13.h}  // 11000000-01000100-10000001-10000000
// CHECK-INST: mov     za0v.h[w12, 0:1], { z12.h, z13.h }
// CHECK-ENCODING: [0x80,0x81,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448180 <unknown>

mov     za0v.h[w14, 2:3], {z0.h, z1.h}  // 11000000-01000100-11000000-00000001
// CHECK-INST: mov     za0v.h[w14, 2:3], { z0.h, z1.h }
// CHECK-ENCODING: [0x01,0xc0,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044c001 <unknown>

mov     za1v.h[w12, 2:3], {z22.h, z23.h}  // 11000000-01000100-10000010-11000101
// CHECK-INST: mov     za1v.h[w12, 2:3], { z22.h, z23.h }
// CHECK-ENCODING: [0xc5,0x82,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c04482c5 <unknown>

mov     za0v.h[w15, 4:5], {z8.h, z9.h}  // 11000000-01000100-11100001-00000010
// CHECK-INST: mov     za0v.h[w15, 4:5], { z8.h, z9.h }
// CHECK-ENCODING: [0x02,0xe1,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044e102 <unknown>

mov     za1v.h[w13, 6:7], {z12.h, z13.h}  // 11000000-01000100-10100001-10000111
// CHECK-INST: mov     za1v.h[w13, 6:7], { z12.h, z13.h }
// CHECK-ENCODING: [0x87,0xa1,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044a187 <unknown>


mova    {z0.s, z1.s}, za0h.s[w12, 0:1]  // 11000000-10000110-00000000-00000000
// CHECK-INST: mov     { z0.s, z1.s }, za0h.s[w12, 0:1]
// CHECK-ENCODING: [0x00,0x00,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0860000 <unknown>

mova    {z20.s, z21.s}, za1h.s[w14, 0:1]  // 11000000-10000110-01000000-01010100
// CHECK-INST: mov     { z20.s, z21.s }, za1h.s[w14, 0:1]
// CHECK-ENCODING: [0x54,0x40,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0864054 <unknown>

mova    {z22.s, z23.s}, za2h.s[w15, 2:3]  // 11000000-10000110-01100000-10110110
// CHECK-INST: mov     { z22.s, z23.s }, za2h.s[w15, 2:3]
// CHECK-ENCODING: [0xb6,0x60,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08660b6 <unknown>

mova    {z30.s, z31.s}, za3h.s[w15, 2:3]  // 11000000-10000110-01100000-11111110
// CHECK-INST: mov     { z30.s, z31.s }, za3h.s[w15, 2:3]
// CHECK-ENCODING: [0xfe,0x60,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08660fe <unknown>

mova    {z4.s, z5.s}, za0h.s[w12, 2:3]  // 11000000-10000110-00000000-00100100
// CHECK-INST: mov     { z4.s, z5.s }, za0h.s[w12, 2:3]
// CHECK-ENCODING: [0x24,0x00,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0860024 <unknown>

mova    {z0.s, z1.s}, za0h.s[w12, 2:3]  // 11000000-10000110-00000000-00100000
// CHECK-INST: mov     { z0.s, z1.s }, za0h.s[w12, 2:3]
// CHECK-ENCODING: [0x20,0x00,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0860020 <unknown>

mova    {z24.s, z25.s}, za1h.s[w14, 2:3]  // 11000000-10000110-01000000-01111000
// CHECK-INST: mov     { z24.s, z25.s }, za1h.s[w14, 2:3]
// CHECK-ENCODING: [0x78,0x40,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0864078 <unknown>

mova    {z0.s, z1.s}, za2h.s[w12, 0:1]  // 11000000-10000110-00000000-10000000
// CHECK-INST: mov     { z0.s, z1.s }, za2h.s[w12, 0:1]
// CHECK-ENCODING: [0x80,0x00,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0860080 <unknown>

mova    {z16.s, z17.s}, za0h.s[w14, 2:3]  // 11000000-10000110-01000000-00110000
// CHECK-INST: mov     { z16.s, z17.s }, za0h.s[w14, 2:3]
// CHECK-ENCODING: [0x30,0x40,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0864030 <unknown>

mova    {z28.s, z29.s}, za3h.s[w12, 0:1]  // 11000000-10000110-00000000-11011100
// CHECK-INST: mov     { z28.s, z29.s }, za3h.s[w12, 0:1]
// CHECK-ENCODING: [0xdc,0x00,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08600dc <unknown>

mova    {z2.s, z3.s}, za0h.s[w15, 2:3]  // 11000000-10000110-01100000-00100010
// CHECK-INST: mov     { z2.s, z3.s }, za0h.s[w15, 2:3]
// CHECK-ENCODING: [0x22,0x60,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0866022 <unknown>

mova    {z6.s, z7.s}, za2h.s[w13, 0:1]  // 11000000-10000110-00100000-10000110
// CHECK-INST: mov     { z6.s, z7.s }, za2h.s[w13, 0:1]
// CHECK-ENCODING: [0x86,0x20,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0862086 <unknown>

// Aliases

mov     {z0.s, z1.s}, za0h.s[w12, 0:1]  // 11000000-10000110-00000000-00000000
// CHECK-INST: mov     { z0.s, z1.s }, za0h.s[w12, 0:1]
// CHECK-ENCODING: [0x00,0x00,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0860000 <unknown>

mov     {z20.s, z21.s}, za1h.s[w14, 0:1]  // 11000000-10000110-01000000-01010100
// CHECK-INST: mov     { z20.s, z21.s }, za1h.s[w14, 0:1]
// CHECK-ENCODING: [0x54,0x40,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0864054 <unknown>

mov     {z22.s, z23.s}, za2h.s[w15, 2:3]  // 11000000-10000110-01100000-10110110
// CHECK-INST: mov     { z22.s, z23.s }, za2h.s[w15, 2:3]
// CHECK-ENCODING: [0xb6,0x60,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08660b6 <unknown>

mov     {z30.s, z31.s}, za3h.s[w15, 2:3]  // 11000000-10000110-01100000-11111110
// CHECK-INST: mov     { z30.s, z31.s }, za3h.s[w15, 2:3]
// CHECK-ENCODING: [0xfe,0x60,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08660fe <unknown>

mov     {z4.s, z5.s}, za0h.s[w12, 2:3]  // 11000000-10000110-00000000-00100100
// CHECK-INST: mov     { z4.s, z5.s }, za0h.s[w12, 2:3]
// CHECK-ENCODING: [0x24,0x00,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0860024 <unknown>

mov     {z0.s, z1.s}, za0h.s[w12, 2:3]  // 11000000-10000110-00000000-00100000
// CHECK-INST: mov     { z0.s, z1.s }, za0h.s[w12, 2:3]
// CHECK-ENCODING: [0x20,0x00,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0860020 <unknown>

mov     {z24.s, z25.s}, za1h.s[w14, 2:3]  // 11000000-10000110-01000000-01111000
// CHECK-INST: mov     { z24.s, z25.s }, za1h.s[w14, 2:3]
// CHECK-ENCODING: [0x78,0x40,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0864078 <unknown>

mov     {z0.s, z1.s}, za2h.s[w12, 0:1]  // 11000000-10000110-00000000-10000000
// CHECK-INST: mov     { z0.s, z1.s }, za2h.s[w12, 0:1]
// CHECK-ENCODING: [0x80,0x00,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0860080 <unknown>

mov     {z16.s, z17.s}, za0h.s[w14, 2:3]  // 11000000-10000110-01000000-00110000
// CHECK-INST: mov     { z16.s, z17.s }, za0h.s[w14, 2:3]
// CHECK-ENCODING: [0x30,0x40,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0864030 <unknown>

mov     {z28.s, z29.s}, za3h.s[w12, 0:1]  // 11000000-10000110-00000000-11011100
// CHECK-INST: mov     { z28.s, z29.s }, za3h.s[w12, 0:1]
// CHECK-ENCODING: [0xdc,0x00,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08600dc <unknown>

mov     {z2.s, z3.s}, za0h.s[w15, 2:3]  // 11000000-10000110-01100000-00100010
// CHECK-INST: mov     { z2.s, z3.s }, za0h.s[w15, 2:3]
// CHECK-ENCODING: [0x22,0x60,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0866022 <unknown>

mov     {z6.s, z7.s}, za2h.s[w13, 0:1]  // 11000000-10000110-00100000-10000110
// CHECK-INST: mov     { z6.s, z7.s }, za2h.s[w13, 0:1]
// CHECK-ENCODING: [0x86,0x20,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0862086 <unknown>


mova    {z0.s, z1.s}, za0v.s[w12, 0:1]  // 11000000-10000110-10000000-00000000
// CHECK-INST: mov     { z0.s, z1.s }, za0v.s[w12, 0:1]
// CHECK-ENCODING: [0x00,0x80,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0868000 <unknown>

mova    {z20.s, z21.s}, za1v.s[w14, 0:1]  // 11000000-10000110-11000000-01010100
// CHECK-INST: mov     { z20.s, z21.s }, za1v.s[w14, 0:1]
// CHECK-ENCODING: [0x54,0xc0,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086c054 <unknown>

mova    {z22.s, z23.s}, za2v.s[w15, 2:3]  // 11000000-10000110-11100000-10110110
// CHECK-INST: mov     { z22.s, z23.s }, za2v.s[w15, 2:3]
// CHECK-ENCODING: [0xb6,0xe0,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086e0b6 <unknown>

mova    {z30.s, z31.s}, za3v.s[w15, 2:3]  // 11000000-10000110-11100000-11111110
// CHECK-INST: mov     { z30.s, z31.s }, za3v.s[w15, 2:3]
// CHECK-ENCODING: [0xfe,0xe0,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086e0fe <unknown>

mova    {z4.s, z5.s}, za0v.s[w12, 2:3]  // 11000000-10000110-10000000-00100100
// CHECK-INST: mov     { z4.s, z5.s }, za0v.s[w12, 2:3]
// CHECK-ENCODING: [0x24,0x80,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0868024 <unknown>

mova    {z0.s, z1.s}, za0v.s[w12, 2:3]  // 11000000-10000110-10000000-00100000
// CHECK-INST: mov     { z0.s, z1.s }, za0v.s[w12, 2:3]
// CHECK-ENCODING: [0x20,0x80,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0868020 <unknown>

mova    {z24.s, z25.s}, za1v.s[w14, 2:3]  // 11000000-10000110-11000000-01111000
// CHECK-INST: mov     { z24.s, z25.s }, za1v.s[w14, 2:3]
// CHECK-ENCODING: [0x78,0xc0,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086c078 <unknown>

mova    {z0.s, z1.s}, za2v.s[w12, 0:1]  // 11000000-10000110-10000000-10000000
// CHECK-INST: mov     { z0.s, z1.s }, za2v.s[w12, 0:1]
// CHECK-ENCODING: [0x80,0x80,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0868080 <unknown>

mova    {z16.s, z17.s}, za0v.s[w14, 2:3]  // 11000000-10000110-11000000-00110000
// CHECK-INST: mov     { z16.s, z17.s }, za0v.s[w14, 2:3]
// CHECK-ENCODING: [0x30,0xc0,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086c030 <unknown>

mova    {z28.s, z29.s}, za3v.s[w12, 0:1]  // 11000000-10000110-10000000-11011100
// CHECK-INST: mov     { z28.s, z29.s }, za3v.s[w12, 0:1]
// CHECK-ENCODING: [0xdc,0x80,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08680dc <unknown>

mova    {z2.s, z3.s}, za0v.s[w15, 2:3]  // 11000000-10000110-11100000-00100010
// CHECK-INST: mov     { z2.s, z3.s }, za0v.s[w15, 2:3]
// CHECK-ENCODING: [0x22,0xe0,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086e022 <unknown>

mova    {z6.s, z7.s}, za2v.s[w13, 0:1]  // 11000000-10000110-10100000-10000110
// CHECK-INST: mov     { z6.s, z7.s }, za2v.s[w13, 0:1]
// CHECK-ENCODING: [0x86,0xa0,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086a086 <unknown>

// Aliases

mov     {z0.s, z1.s}, za0v.s[w12, 0:1]  // 11000000-10000110-10000000-00000000
// CHECK-INST: mov     { z0.s, z1.s }, za0v.s[w12, 0:1]
// CHECK-ENCODING: [0x00,0x80,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0868000 <unknown>

mov     {z20.s, z21.s}, za1v.s[w14, 0:1]  // 11000000-10000110-11000000-01010100
// CHECK-INST: mov     { z20.s, z21.s }, za1v.s[w14, 0:1]
// CHECK-ENCODING: [0x54,0xc0,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086c054 <unknown>

mov     {z22.s, z23.s}, za2v.s[w15, 2:3]  // 11000000-10000110-11100000-10110110
// CHECK-INST: mov     { z22.s, z23.s }, za2v.s[w15, 2:3]
// CHECK-ENCODING: [0xb6,0xe0,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086e0b6 <unknown>

mov     {z30.s, z31.s}, za3v.s[w15, 2:3]  // 11000000-10000110-11100000-11111110
// CHECK-INST: mov     { z30.s, z31.s }, za3v.s[w15, 2:3]
// CHECK-ENCODING: [0xfe,0xe0,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086e0fe <unknown>

mov     {z4.s, z5.s}, za0v.s[w12, 2:3]  // 11000000-10000110-10000000-00100100
// CHECK-INST: mov     { z4.s, z5.s }, za0v.s[w12, 2:3]
// CHECK-ENCODING: [0x24,0x80,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0868024 <unknown>

mov     {z0.s, z1.s}, za0v.s[w12, 2:3]  // 11000000-10000110-10000000-00100000
// CHECK-INST: mov     { z0.s, z1.s }, za0v.s[w12, 2:3]
// CHECK-ENCODING: [0x20,0x80,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0868020 <unknown>

mov     {z24.s, z25.s}, za1v.s[w14, 2:3]  // 11000000-10000110-11000000-01111000
// CHECK-INST: mov     { z24.s, z25.s }, za1v.s[w14, 2:3]
// CHECK-ENCODING: [0x78,0xc0,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086c078 <unknown>

mov     {z0.s, z1.s}, za2v.s[w12, 0:1]  // 11000000-10000110-10000000-10000000
// CHECK-INST: mov     { z0.s, z1.s }, za2v.s[w12, 0:1]
// CHECK-ENCODING: [0x80,0x80,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0868080 <unknown>

mov     {z16.s, z17.s}, za0v.s[w14, 2:3]  // 11000000-10000110-11000000-00110000
// CHECK-INST: mov     { z16.s, z17.s }, za0v.s[w14, 2:3]
// CHECK-ENCODING: [0x30,0xc0,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086c030 <unknown>

mov     {z28.s, z29.s}, za3v.s[w12, 0:1]  // 11000000-10000110-10000000-11011100
// CHECK-INST: mov     { z28.s, z29.s }, za3v.s[w12, 0:1]
// CHECK-ENCODING: [0xdc,0x80,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08680dc <unknown>

mov     {z2.s, z3.s}, za0v.s[w15, 2:3]  // 11000000-10000110-11100000-00100010
// CHECK-INST: mov     { z2.s, z3.s }, za0v.s[w15, 2:3]
// CHECK-ENCODING: [0x22,0xe0,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086e022 <unknown>

mov     {z6.s, z7.s}, za2v.s[w13, 0:1]  // 11000000-10000110-10100000-10000110
// CHECK-INST: mov     { z6.s, z7.s }, za2v.s[w13, 0:1]
// CHECK-ENCODING: [0x86,0xa0,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086a086 <unknown>


mova    za0h.s[w12, 0:1], {z0.s, z1.s}  // 11000000-10000100-00000000-00000000
// CHECK-INST: mov     za0h.s[w12, 0:1], { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0x00,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840000 <unknown>

mova    za2h.s[w14, 2:3], {z10.s, z11.s}  // 11000000-10000100-01000001-01000101
// CHECK-INST: mov     za2h.s[w14, 2:3], { z10.s, z11.s }
// CHECK-ENCODING: [0x45,0x41,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0844145 <unknown>

mova    za3h.s[w15, 2:3], {z12.s, z13.s}  // 11000000-10000100-01100001-10000111
// CHECK-INST: mov     za3h.s[w15, 2:3], { z12.s, z13.s }
// CHECK-ENCODING: [0x87,0x61,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0846187 <unknown>

mova    za3h.s[w15, 2:3], {z30.s, z31.s}  // 11000000-10000100-01100011-11000111
// CHECK-INST: mov     za3h.s[w15, 2:3], { z30.s, z31.s }
// CHECK-ENCODING: [0xc7,0x63,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08463c7 <unknown>

mova    za2h.s[w12, 2:3], {z16.s, z17.s}  // 11000000-10000100-00000010-00000101
// CHECK-INST: mov     za2h.s[w12, 2:3], { z16.s, z17.s }
// CHECK-ENCODING: [0x05,0x02,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840205 <unknown>

mova    za0h.s[w12, 2:3], {z0.s, z1.s}  // 11000000-10000100-00000000-00000001
// CHECK-INST: mov     za0h.s[w12, 2:3], { z0.s, z1.s }
// CHECK-ENCODING: [0x01,0x00,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840001 <unknown>

mova    za0h.s[w14, 0:1], {z18.s, z19.s}  // 11000000-10000100-01000010-01000000
// CHECK-INST: mov     za0h.s[w14, 0:1], { z18.s, z19.s }
// CHECK-ENCODING: [0x40,0x42,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0844240 <unknown>

mova    za0h.s[w12, 0:1], {z12.s, z13.s}  // 11000000-10000100-00000001-10000000
// CHECK-INST: mov     za0h.s[w12, 0:1], { z12.s, z13.s }
// CHECK-ENCODING: [0x80,0x01,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840180 <unknown>

mova    za0h.s[w14, 2:3], {z0.s, z1.s}  // 11000000-10000100-01000000-00000001
// CHECK-INST: mov     za0h.s[w14, 2:3], { z0.s, z1.s }
// CHECK-ENCODING: [0x01,0x40,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0844001 <unknown>

mova    za2h.s[w12, 2:3], {z22.s, z23.s}  // 11000000-10000100-00000010-11000101
// CHECK-INST: mov     za2h.s[w12, 2:3], { z22.s, z23.s }
// CHECK-ENCODING: [0xc5,0x02,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08402c5 <unknown>

mova    za1h.s[w15, 0:1], {z8.s, z9.s}  // 11000000-10000100-01100001-00000010
// CHECK-INST: mov     za1h.s[w15, 0:1], { z8.s, z9.s }
// CHECK-ENCODING: [0x02,0x61,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0846102 <unknown>

mova    za3h.s[w13, 2:3], {z12.s, z13.s}  // 11000000-10000100-00100001-10000111
// CHECK-INST: mov     za3h.s[w13, 2:3], { z12.s, z13.s }
// CHECK-ENCODING: [0x87,0x21,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0842187 <unknown>

// Aliases

mov     za0h.s[w12, 0:1], {z0.s, z1.s}  // 11000000-10000100-00000000-00000000
// CHECK-INST: mov     za0h.s[w12, 0:1], { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0x00,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840000 <unknown>

mov     za2h.s[w14, 2:3], {z10.s, z11.s}  // 11000000-10000100-01000001-01000101
// CHECK-INST: mov     za2h.s[w14, 2:3], { z10.s, z11.s }
// CHECK-ENCODING: [0x45,0x41,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0844145 <unknown>

mov     za3h.s[w15, 2:3], {z12.s, z13.s}  // 11000000-10000100-01100001-10000111
// CHECK-INST: mov     za3h.s[w15, 2:3], { z12.s, z13.s }
// CHECK-ENCODING: [0x87,0x61,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0846187 <unknown>

mov     za3h.s[w15, 2:3], {z30.s, z31.s}  // 11000000-10000100-01100011-11000111
// CHECK-INST: mov     za3h.s[w15, 2:3], { z30.s, z31.s }
// CHECK-ENCODING: [0xc7,0x63,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08463c7 <unknown>

mov     za2h.s[w12, 2:3], {z16.s, z17.s}  // 11000000-10000100-00000010-00000101
// CHECK-INST: mov     za2h.s[w12, 2:3], { z16.s, z17.s }
// CHECK-ENCODING: [0x05,0x02,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840205 <unknown>

mov     za0h.s[w12, 2:3], {z0.s, z1.s}  // 11000000-10000100-00000000-00000001
// CHECK-INST: mov     za0h.s[w12, 2:3], { z0.s, z1.s }
// CHECK-ENCODING: [0x01,0x00,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840001 <unknown>

mov     za0h.s[w14, 0:1], {z18.s, z19.s}  // 11000000-10000100-01000010-01000000
// CHECK-INST: mov     za0h.s[w14, 0:1], { z18.s, z19.s }
// CHECK-ENCODING: [0x40,0x42,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0844240 <unknown>

mov     za0h.s[w12, 0:1], {z12.s, z13.s}  // 11000000-10000100-00000001-10000000
// CHECK-INST: mov     za0h.s[w12, 0:1], { z12.s, z13.s }
// CHECK-ENCODING: [0x80,0x01,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840180 <unknown>

mov     za0h.s[w14, 2:3], {z0.s, z1.s}  // 11000000-10000100-01000000-00000001
// CHECK-INST: mov     za0h.s[w14, 2:3], { z0.s, z1.s }
// CHECK-ENCODING: [0x01,0x40,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0844001 <unknown>

mov     za2h.s[w12, 2:3], {z22.s, z23.s}  // 11000000-10000100-00000010-11000101
// CHECK-INST: mov     za2h.s[w12, 2:3], { z22.s, z23.s }
// CHECK-ENCODING: [0xc5,0x02,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08402c5 <unknown>

mov     za1h.s[w15, 0:1], {z8.s, z9.s}  // 11000000-10000100-01100001-00000010
// CHECK-INST: mov     za1h.s[w15, 0:1], { z8.s, z9.s }
// CHECK-ENCODING: [0x02,0x61,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0846102 <unknown>

mov     za3h.s[w13, 2:3], {z12.s, z13.s}  // 11000000-10000100-00100001-10000111
// CHECK-INST: mov     za3h.s[w13, 2:3], { z12.s, z13.s }
// CHECK-ENCODING: [0x87,0x21,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0842187 <unknown>


mova    za0v.s[w12, 0:1], {z0.s, z1.s}  // 11000000-10000100-10000000-00000000
// CHECK-INST: mov     za0v.s[w12, 0:1], { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0x80,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848000 <unknown>

mova    za2v.s[w14, 2:3], {z10.s, z11.s}  // 11000000-10000100-11000001-01000101
// CHECK-INST: mov     za2v.s[w14, 2:3], { z10.s, z11.s }
// CHECK-ENCODING: [0x45,0xc1,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084c145 <unknown>

mova    za3v.s[w15, 2:3], {z12.s, z13.s}  // 11000000-10000100-11100001-10000111
// CHECK-INST: mov     za3v.s[w15, 2:3], { z12.s, z13.s }
// CHECK-ENCODING: [0x87,0xe1,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084e187 <unknown>

mova    za3v.s[w15, 2:3], {z30.s, z31.s}  // 11000000-10000100-11100011-11000111
// CHECK-INST: mov     za3v.s[w15, 2:3], { z30.s, z31.s }
// CHECK-ENCODING: [0xc7,0xe3,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084e3c7 <unknown>

mova    za2v.s[w12, 2:3], {z16.s, z17.s}  // 11000000-10000100-10000010-00000101
// CHECK-INST: mov     za2v.s[w12, 2:3], { z16.s, z17.s }
// CHECK-ENCODING: [0x05,0x82,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848205 <unknown>

mova    za0v.s[w12, 2:3], {z0.s, z1.s}  // 11000000-10000100-10000000-00000001
// CHECK-INST: mov     za0v.s[w12, 2:3], { z0.s, z1.s }
// CHECK-ENCODING: [0x01,0x80,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848001 <unknown>

mova    za0v.s[w14, 0:1], {z18.s, z19.s}  // 11000000-10000100-11000010-01000000
// CHECK-INST: mov     za0v.s[w14, 0:1], { z18.s, z19.s }
// CHECK-ENCODING: [0x40,0xc2,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084c240 <unknown>

mova    za0v.s[w12, 0:1], {z12.s, z13.s}  // 11000000-10000100-10000001-10000000
// CHECK-INST: mov     za0v.s[w12, 0:1], { z12.s, z13.s }
// CHECK-ENCODING: [0x80,0x81,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848180 <unknown>

mova    za0v.s[w14, 2:3], {z0.s, z1.s}  // 11000000-10000100-11000000-00000001
// CHECK-INST: mov     za0v.s[w14, 2:3], { z0.s, z1.s }
// CHECK-ENCODING: [0x01,0xc0,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084c001 <unknown>

mova    za2v.s[w12, 2:3], {z22.s, z23.s}  // 11000000-10000100-10000010-11000101
// CHECK-INST: mov     za2v.s[w12, 2:3], { z22.s, z23.s }
// CHECK-ENCODING: [0xc5,0x82,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08482c5 <unknown>

mova    za1v.s[w15, 0:1], {z8.s, z9.s}  // 11000000-10000100-11100001-00000010
// CHECK-INST: mov     za1v.s[w15, 0:1], { z8.s, z9.s }
// CHECK-ENCODING: [0x02,0xe1,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084e102 <unknown>

mova    za3v.s[w13, 2:3], {z12.s, z13.s}  // 11000000-10000100-10100001-10000111
// CHECK-INST: mov     za3v.s[w13, 2:3], { z12.s, z13.s }
// CHECK-ENCODING: [0x87,0xa1,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084a187 <unknown>

// Aliases

mov     za0v.s[w12, 0:1], {z0.s, z1.s}  // 11000000-10000100-10000000-00000000
// CHECK-INST: mov     za0v.s[w12, 0:1], { z0.s, z1.s }
// CHECK-ENCODING: [0x00,0x80,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848000 <unknown>

mov     za2v.s[w14, 2:3], {z10.s, z11.s}  // 11000000-10000100-11000001-01000101
// CHECK-INST: mov     za2v.s[w14, 2:3], { z10.s, z11.s }
// CHECK-ENCODING: [0x45,0xc1,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084c145 <unknown>

mov     za3v.s[w15, 2:3], {z12.s, z13.s}  // 11000000-10000100-11100001-10000111
// CHECK-INST: mov     za3v.s[w15, 2:3], { z12.s, z13.s }
// CHECK-ENCODING: [0x87,0xe1,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084e187 <unknown>

mov     za3v.s[w15, 2:3], {z30.s, z31.s}  // 11000000-10000100-11100011-11000111
// CHECK-INST: mov     za3v.s[w15, 2:3], { z30.s, z31.s }
// CHECK-ENCODING: [0xc7,0xe3,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084e3c7 <unknown>

mov     za2v.s[w12, 2:3], {z16.s, z17.s}  // 11000000-10000100-10000010-00000101
// CHECK-INST: mov     za2v.s[w12, 2:3], { z16.s, z17.s }
// CHECK-ENCODING: [0x05,0x82,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848205 <unknown>

mov     za0v.s[w12, 2:3], {z0.s, z1.s}  // 11000000-10000100-10000000-00000001
// CHECK-INST: mov     za0v.s[w12, 2:3], { z0.s, z1.s }
// CHECK-ENCODING: [0x01,0x80,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848001 <unknown>

mov     za0v.s[w14, 0:1], {z18.s, z19.s}  // 11000000-10000100-11000010-01000000
// CHECK-INST: mov     za0v.s[w14, 0:1], { z18.s, z19.s }
// CHECK-ENCODING: [0x40,0xc2,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084c240 <unknown>

mov     za0v.s[w12, 0:1], {z12.s, z13.s}  // 11000000-10000100-10000001-10000000
// CHECK-INST: mov     za0v.s[w12, 0:1], { z12.s, z13.s }
// CHECK-ENCODING: [0x80,0x81,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848180 <unknown>

mov     za0v.s[w14, 2:3], {z0.s, z1.s}  // 11000000-10000100-11000000-00000001
// CHECK-INST: mov     za0v.s[w14, 2:3], { z0.s, z1.s }
// CHECK-ENCODING: [0x01,0xc0,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084c001 <unknown>

mov     za2v.s[w12, 2:3], {z22.s, z23.s}  // 11000000-10000100-10000010-11000101
// CHECK-INST: mov     za2v.s[w12, 2:3], { z22.s, z23.s }
// CHECK-ENCODING: [0xc5,0x82,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c08482c5 <unknown>

mov     za1v.s[w15, 0:1], {z8.s, z9.s}  // 11000000-10000100-11100001-00000010
// CHECK-INST: mov     za1v.s[w15, 0:1], { z8.s, z9.s }
// CHECK-ENCODING: [0x02,0xe1,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084e102 <unknown>

mov     za3v.s[w13, 2:3], {z12.s, z13.s}  // 11000000-10000100-10100001-10000111
// CHECK-INST: mov     za3v.s[w13, 2:3], { z12.s, z13.s }
// CHECK-ENCODING: [0x87,0xa1,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084a187 <unknown>


mova    {z0.d, z1.d}, za0h.d[w12, 0:1]  // 11000000-11000110-00000000-00000000
// CHECK-INST: mov     { z0.d, z1.d }, za0h.d[w12, 0:1]
// CHECK-ENCODING: [0x00,0x00,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c60000 <unknown>

mova    {z20.d, z21.d}, za2h.d[w14, 0:1]  // 11000000-11000110-01000000-01010100
// CHECK-INST: mov     { z20.d, z21.d }, za2h.d[w14, 0:1]
// CHECK-ENCODING: [0x54,0x40,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c64054 <unknown>

mova    {z22.d, z23.d}, za5h.d[w15, 0:1]  // 11000000-11000110-01100000-10110110
// CHECK-INST: mov     { z22.d, z23.d }, za5h.d[w15, 0:1]
// CHECK-ENCODING: [0xb6,0x60,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c660b6 <unknown>

mova    {z30.d, z31.d}, za7h.d[w15, 0:1]  // 11000000-11000110-01100000-11111110
// CHECK-INST: mov     { z30.d, z31.d }, za7h.d[w15, 0:1]
// CHECK-ENCODING: [0xfe,0x60,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c660fe <unknown>

mova    {z4.d, z5.d}, za1h.d[w12, 0:1]  // 11000000-11000110-00000000-00100100
// CHECK-INST: mov     { z4.d, z5.d }, za1h.d[w12, 0:1]
// CHECK-ENCODING: [0x24,0x00,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c60024 <unknown>

mova    {z0.d, z1.d}, za1h.d[w12, 0:1]  // 11000000-11000110-00000000-00100000
// CHECK-INST: mov     { z0.d, z1.d }, za1h.d[w12, 0:1]
// CHECK-ENCODING: [0x20,0x00,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c60020 <unknown>

mova    {z24.d, z25.d}, za3h.d[w14, 0:1]  // 11000000-11000110-01000000-01111000
// CHECK-INST: mov     { z24.d, z25.d }, za3h.d[w14, 0:1]
// CHECK-ENCODING: [0x78,0x40,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c64078 <unknown>

mova    {z0.d, z1.d}, za4h.d[w12, 0:1]  // 11000000-11000110-00000000-10000000
// CHECK-INST: mov     { z0.d, z1.d }, za4h.d[w12, 0:1]
// CHECK-ENCODING: [0x80,0x00,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c60080 <unknown>

mova    {z16.d, z17.d}, za1h.d[w14, 0:1]  // 11000000-11000110-01000000-00110000
// CHECK-INST: mov     { z16.d, z17.d }, za1h.d[w14, 0:1]
// CHECK-ENCODING: [0x30,0x40,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c64030 <unknown>

mova    {z28.d, z29.d}, za6h.d[w12, 0:1]  // 11000000-11000110-00000000-11011100
// CHECK-INST: mov     { z28.d, z29.d }, za6h.d[w12, 0:1]
// CHECK-ENCODING: [0xdc,0x00,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c600dc <unknown>

mova    {z2.d, z3.d}, za1h.d[w15, 0:1]  // 11000000-11000110-01100000-00100010
// CHECK-INST: mov     { z2.d, z3.d }, za1h.d[w15, 0:1]
// CHECK-ENCODING: [0x22,0x60,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c66022 <unknown>

mova    {z6.d, z7.d}, za4h.d[w13, 0:1]  // 11000000-11000110-00100000-10000110
// CHECK-INST: mov     { z6.d, z7.d }, za4h.d[w13, 0:1]
// CHECK-ENCODING: [0x86,0x20,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c62086 <unknown>

// Aliases

mov     {z0.d, z1.d}, za0h.d[w12, 0:1]  // 11000000-11000110-00000000-00000000
// CHECK-INST: mov     { z0.d, z1.d }, za0h.d[w12, 0:1]
// CHECK-ENCODING: [0x00,0x00,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c60000 <unknown>

mov     {z20.d, z21.d}, za2h.d[w14, 0:1]  // 11000000-11000110-01000000-01010100
// CHECK-INST: mov     { z20.d, z21.d }, za2h.d[w14, 0:1]
// CHECK-ENCODING: [0x54,0x40,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c64054 <unknown>

mov     {z22.d, z23.d}, za5h.d[w15, 0:1]  // 11000000-11000110-01100000-10110110
// CHECK-INST: mov     { z22.d, z23.d }, za5h.d[w15, 0:1]
// CHECK-ENCODING: [0xb6,0x60,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c660b6 <unknown>

mov     {z30.d, z31.d}, za7h.d[w15, 0:1]  // 11000000-11000110-01100000-11111110
// CHECK-INST: mov     { z30.d, z31.d }, za7h.d[w15, 0:1]
// CHECK-ENCODING: [0xfe,0x60,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c660fe <unknown>

mov     {z4.d, z5.d}, za1h.d[w12, 0:1]  // 11000000-11000110-00000000-00100100
// CHECK-INST: mov     { z4.d, z5.d }, za1h.d[w12, 0:1]
// CHECK-ENCODING: [0x24,0x00,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c60024 <unknown>

mov     {z0.d, z1.d}, za1h.d[w12, 0:1]  // 11000000-11000110-00000000-00100000
// CHECK-INST: mov     { z0.d, z1.d }, za1h.d[w12, 0:1]
// CHECK-ENCODING: [0x20,0x00,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c60020 <unknown>

mov     {z24.d, z25.d}, za3h.d[w14, 0:1]  // 11000000-11000110-01000000-01111000
// CHECK-INST: mov     { z24.d, z25.d }, za3h.d[w14, 0:1]
// CHECK-ENCODING: [0x78,0x40,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c64078 <unknown>

mov     {z0.d, z1.d}, za4h.d[w12, 0:1]  // 11000000-11000110-00000000-10000000
// CHECK-INST: mov     { z0.d, z1.d }, za4h.d[w12, 0:1]
// CHECK-ENCODING: [0x80,0x00,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c60080 <unknown>

mov     {z16.d, z17.d}, za1h.d[w14, 0:1]  // 11000000-11000110-01000000-00110000
// CHECK-INST: mov     { z16.d, z17.d }, za1h.d[w14, 0:1]
// CHECK-ENCODING: [0x30,0x40,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c64030 <unknown>

mov     {z28.d, z29.d}, za6h.d[w12, 0:1]  // 11000000-11000110-00000000-11011100
// CHECK-INST: mov     { z28.d, z29.d }, za6h.d[w12, 0:1]
// CHECK-ENCODING: [0xdc,0x00,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c600dc <unknown>

mov     {z2.d, z3.d}, za1h.d[w15, 0:1]  // 11000000-11000110-01100000-00100010
// CHECK-INST: mov     { z2.d, z3.d }, za1h.d[w15, 0:1]
// CHECK-ENCODING: [0x22,0x60,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c66022 <unknown>

mov     {z6.d, z7.d}, za4h.d[w13, 0:1]  // 11000000-11000110-00100000-10000110
// CHECK-INST: mov     { z6.d, z7.d }, za4h.d[w13, 0:1]
// CHECK-ENCODING: [0x86,0x20,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c62086 <unknown>


mova    {z0.d, z1.d}, za0v.d[w12, 0:1]  // 11000000-11000110-10000000-00000000
// CHECK-INST: mov     { z0.d, z1.d }, za0v.d[w12, 0:1]
// CHECK-ENCODING: [0x00,0x80,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c68000 <unknown>

mova    {z20.d, z21.d}, za2v.d[w14, 0:1]  // 11000000-11000110-11000000-01010100
// CHECK-INST: mov     { z20.d, z21.d }, za2v.d[w14, 0:1]
// CHECK-ENCODING: [0x54,0xc0,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6c054 <unknown>

mova    {z22.d, z23.d}, za5v.d[w15, 0:1]  // 11000000-11000110-11100000-10110110
// CHECK-INST: mov     { z22.d, z23.d }, za5v.d[w15, 0:1]
// CHECK-ENCODING: [0xb6,0xe0,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6e0b6 <unknown>

mova    {z30.d, z31.d}, za7v.d[w15, 0:1]  // 11000000-11000110-11100000-11111110
// CHECK-INST: mov     { z30.d, z31.d }, za7v.d[w15, 0:1]
// CHECK-ENCODING: [0xfe,0xe0,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6e0fe <unknown>

mova    {z4.d, z5.d}, za1v.d[w12, 0:1]  // 11000000-11000110-10000000-00100100
// CHECK-INST: mov     { z4.d, z5.d }, za1v.d[w12, 0:1]
// CHECK-ENCODING: [0x24,0x80,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c68024 <unknown>

mova    {z0.d, z1.d}, za1v.d[w12, 0:1]  // 11000000-11000110-10000000-00100000
// CHECK-INST: mov     { z0.d, z1.d }, za1v.d[w12, 0:1]
// CHECK-ENCODING: [0x20,0x80,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c68020 <unknown>

mova    {z24.d, z25.d}, za3v.d[w14, 0:1]  // 11000000-11000110-11000000-01111000
// CHECK-INST: mov     { z24.d, z25.d }, za3v.d[w14, 0:1]
// CHECK-ENCODING: [0x78,0xc0,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6c078 <unknown>

mova    {z0.d, z1.d}, za4v.d[w12, 0:1]  // 11000000-11000110-10000000-10000000
// CHECK-INST: mov     { z0.d, z1.d }, za4v.d[w12, 0:1]
// CHECK-ENCODING: [0x80,0x80,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c68080 <unknown>

mova    {z16.d, z17.d}, za1v.d[w14, 0:1]  // 11000000-11000110-11000000-00110000
// CHECK-INST: mov     { z16.d, z17.d }, za1v.d[w14, 0:1]
// CHECK-ENCODING: [0x30,0xc0,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6c030 <unknown>

mova    {z28.d, z29.d}, za6v.d[w12, 0:1]  // 11000000-11000110-10000000-11011100
// CHECK-INST: mov     { z28.d, z29.d }, za6v.d[w12, 0:1]
// CHECK-ENCODING: [0xdc,0x80,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c680dc <unknown>

mova    {z2.d, z3.d}, za1v.d[w15, 0:1]  // 11000000-11000110-11100000-00100010
// CHECK-INST: mov     { z2.d, z3.d }, za1v.d[w15, 0:1]
// CHECK-ENCODING: [0x22,0xe0,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6e022 <unknown>

mova    {z6.d, z7.d}, za4v.d[w13, 0:1]  // 11000000-11000110-10100000-10000110
// CHECK-INST: mov     { z6.d, z7.d }, za4v.d[w13, 0:1]
// CHECK-ENCODING: [0x86,0xa0,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6a086 <unknown>

// Aliases

mov     {z0.d, z1.d}, za0v.d[w12, 0:1]  // 11000000-11000110-10000000-00000000
// CHECK-INST: mov     { z0.d, z1.d }, za0v.d[w12, 0:1]
// CHECK-ENCODING: [0x00,0x80,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c68000 <unknown>

mov     {z20.d, z21.d}, za2v.d[w14, 0:1]  // 11000000-11000110-11000000-01010100
// CHECK-INST: mov     { z20.d, z21.d }, za2v.d[w14, 0:1]
// CHECK-ENCODING: [0x54,0xc0,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6c054 <unknown>

mov     {z22.d, z23.d}, za5v.d[w15, 0:1]  // 11000000-11000110-11100000-10110110
// CHECK-INST: mov     { z22.d, z23.d }, za5v.d[w15, 0:1]
// CHECK-ENCODING: [0xb6,0xe0,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6e0b6 <unknown>

mov     {z30.d, z31.d}, za7v.d[w15, 0:1]  // 11000000-11000110-11100000-11111110
// CHECK-INST: mov     { z30.d, z31.d }, za7v.d[w15, 0:1]
// CHECK-ENCODING: [0xfe,0xe0,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6e0fe <unknown>

mov     {z4.d, z5.d}, za1v.d[w12, 0:1]  // 11000000-11000110-10000000-00100100
// CHECK-INST: mov     { z4.d, z5.d }, za1v.d[w12, 0:1]
// CHECK-ENCODING: [0x24,0x80,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c68024 <unknown>

mov     {z0.d, z1.d}, za1v.d[w12, 0:1]  // 11000000-11000110-10000000-00100000
// CHECK-INST: mov     { z0.d, z1.d }, za1v.d[w12, 0:1]
// CHECK-ENCODING: [0x20,0x80,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c68020 <unknown>

mov     {z24.d, z25.d}, za3v.d[w14, 0:1]  // 11000000-11000110-11000000-01111000
// CHECK-INST: mov     { z24.d, z25.d }, za3v.d[w14, 0:1]
// CHECK-ENCODING: [0x78,0xc0,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6c078 <unknown>

mov     {z0.d, z1.d}, za4v.d[w12, 0:1]  // 11000000-11000110-10000000-10000000
// CHECK-INST: mov     { z0.d, z1.d }, za4v.d[w12, 0:1]
// CHECK-ENCODING: [0x80,0x80,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c68080 <unknown>

mov     {z16.d, z17.d}, za1v.d[w14, 0:1]  // 11000000-11000110-11000000-00110000
// CHECK-INST: mov     { z16.d, z17.d }, za1v.d[w14, 0:1]
// CHECK-ENCODING: [0x30,0xc0,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6c030 <unknown>

mov     {z28.d, z29.d}, za6v.d[w12, 0:1]  // 11000000-11000110-10000000-11011100
// CHECK-INST: mov     { z28.d, z29.d }, za6v.d[w12, 0:1]
// CHECK-ENCODING: [0xdc,0x80,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c680dc <unknown>

mov     {z2.d, z3.d}, za1v.d[w15, 0:1]  // 11000000-11000110-11100000-00100010
// CHECK-INST: mov     { z2.d, z3.d }, za1v.d[w15, 0:1]
// CHECK-ENCODING: [0x22,0xe0,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6e022 <unknown>

mov     {z6.d, z7.d}, za4v.d[w13, 0:1]  // 11000000-11000110-10100000-10000110
// CHECK-INST: mov     { z6.d, z7.d }, za4v.d[w13, 0:1]
// CHECK-ENCODING: [0x86,0xa0,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6a086 <unknown>


mova    {z0.d, z1.d}, za.d[w8, 0, vgx2]  // 11000000-00000110-00001000-00000000
// CHECK-INST: mov     { z0.d, z1.d }, za.d[w8, 0, vgx2]
// CHECK-ENCODING: [0x00,0x08,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060800 <unknown>

mova    {z0.d, z1.d}, za.d[w8, 0]  // 11000000-00000110-00001000-00000000
// CHECK-INST: mov     { z0.d, z1.d }, za.d[w8, 0, vgx2]
// CHECK-ENCODING: [0x00,0x08,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060800 <unknown>

mova    {z20.d, z21.d}, za.d[w10, 2, vgx2]  // 11000000-00000110-01001000-01010100
// CHECK-INST: mov     { z20.d, z21.d }, za.d[w10, 2, vgx2]
// CHECK-ENCODING: [0x54,0x48,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064854 <unknown>

mova    {z20.d, z21.d}, za.d[w10, 2]  // 11000000-00000110-01001000-01010100
// CHECK-INST: mov     { z20.d, z21.d }, za.d[w10, 2, vgx2]
// CHECK-ENCODING: [0x54,0x48,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064854 <unknown>

mova    {z22.d, z23.d}, za.d[w11, 5, vgx2]  // 11000000-00000110-01101000-10110110
// CHECK-INST: mov     { z22.d, z23.d }, za.d[w11, 5, vgx2]
// CHECK-ENCODING: [0xb6,0x68,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00668b6 <unknown>

mova    {z22.d, z23.d}, za.d[w11, 5]  // 11000000-00000110-01101000-10110110
// CHECK-INST: mov     { z22.d, z23.d }, za.d[w11, 5, vgx2]
// CHECK-ENCODING: [0xb6,0x68,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00668b6 <unknown>

mova    {z30.d, z31.d}, za.d[w11, 7, vgx2]  // 11000000-00000110-01101000-11111110
// CHECK-INST: mov     { z30.d, z31.d }, za.d[w11, 7, vgx2]
// CHECK-ENCODING: [0xfe,0x68,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00668fe <unknown>

mova    {z30.d, z31.d}, za.d[w11, 7]  // 11000000-00000110-01101000-11111110
// CHECK-INST: mov     { z30.d, z31.d }, za.d[w11, 7, vgx2]
// CHECK-ENCODING: [0xfe,0x68,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00668fe <unknown>

mova    {z4.d, z5.d}, za.d[w8, 1, vgx2]  // 11000000-00000110-00001000-00100100
// CHECK-INST: mov     { z4.d, z5.d }, za.d[w8, 1, vgx2]
// CHECK-ENCODING: [0x24,0x08,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060824 <unknown>

mova    {z4.d, z5.d}, za.d[w8, 1]  // 11000000-00000110-00001000-00100100
// CHECK-INST: mov     { z4.d, z5.d }, za.d[w8, 1, vgx2]
// CHECK-ENCODING: [0x24,0x08,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060824 <unknown>

mova    {z0.d, z1.d}, za.d[w8, 1, vgx2]  // 11000000-00000110-00001000-00100000
// CHECK-INST: mov     { z0.d, z1.d }, za.d[w8, 1, vgx2]
// CHECK-ENCODING: [0x20,0x08,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060820 <unknown>

mova    {z0.d, z1.d}, za.d[w8, 1]  // 11000000-00000110-00001000-00100000
// CHECK-INST: mov     { z0.d, z1.d }, za.d[w8, 1, vgx2]
// CHECK-ENCODING: [0x20,0x08,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060820 <unknown>

mova    {z24.d, z25.d}, za.d[w10, 3, vgx2]  // 11000000-00000110-01001000-01111000
// CHECK-INST: mov     { z24.d, z25.d }, za.d[w10, 3, vgx2]
// CHECK-ENCODING: [0x78,0x48,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064878 <unknown>

mova    {z24.d, z25.d}, za.d[w10, 3]  // 11000000-00000110-01001000-01111000
// CHECK-INST: mov     { z24.d, z25.d }, za.d[w10, 3, vgx2]
// CHECK-ENCODING: [0x78,0x48,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064878 <unknown>

mova    {z0.d, z1.d}, za.d[w8, 4, vgx2]  // 11000000-00000110-00001000-10000000
// CHECK-INST: mov     { z0.d, z1.d }, za.d[w8, 4, vgx2]
// CHECK-ENCODING: [0x80,0x08,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060880 <unknown>

mova    {z0.d, z1.d}, za.d[w8, 4]  // 11000000-00000110-00001000-10000000
// CHECK-INST: mov     { z0.d, z1.d }, za.d[w8, 4, vgx2]
// CHECK-ENCODING: [0x80,0x08,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060880 <unknown>

mova    {z16.d, z17.d}, za.d[w10, 1, vgx2]  // 11000000-00000110-01001000-00110000
// CHECK-INST: mov     { z16.d, z17.d }, za.d[w10, 1, vgx2]
// CHECK-ENCODING: [0x30,0x48,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064830 <unknown>

mova    {z16.d, z17.d}, za.d[w10, 1]  // 11000000-00000110-01001000-00110000
// CHECK-INST: mov     { z16.d, z17.d }, za.d[w10, 1, vgx2]
// CHECK-ENCODING: [0x30,0x48,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064830 <unknown>

mova    {z28.d, z29.d}, za.d[w8, 6, vgx2]  // 11000000-00000110-00001000-11011100
// CHECK-INST: mov     { z28.d, z29.d }, za.d[w8, 6, vgx2]
// CHECK-ENCODING: [0xdc,0x08,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00608dc <unknown>

mova    {z28.d, z29.d}, za.d[w8, 6]  // 11000000-00000110-00001000-11011100
// CHECK-INST: mov     { z28.d, z29.d }, za.d[w8, 6, vgx2]
// CHECK-ENCODING: [0xdc,0x08,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00608dc <unknown>

mova    {z2.d, z3.d}, za.d[w11, 1, vgx2]  // 11000000-00000110-01101000-00100010
// CHECK-INST: mov     { z2.d, z3.d }, za.d[w11, 1, vgx2]
// CHECK-ENCODING: [0x22,0x68,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066822 <unknown>

mova    {z2.d, z3.d}, za.d[w11, 1]  // 11000000-00000110-01101000-00100010
// CHECK-INST: mov     { z2.d, z3.d }, za.d[w11, 1, vgx2]
// CHECK-ENCODING: [0x22,0x68,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066822 <unknown>

mova    {z6.d, z7.d}, za.d[w9, 4, vgx2]  // 11000000-00000110-00101000-10000110
// CHECK-INST: mov     { z6.d, z7.d }, za.d[w9, 4, vgx2]
// CHECK-ENCODING: [0x86,0x28,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0062886 <unknown>

mova    {z6.d, z7.d}, za.d[w9, 4]  // 11000000-00000110-00101000-10000110
// CHECK-INST: mov     { z6.d, z7.d }, za.d[w9, 4, vgx2]
// CHECK-ENCODING: [0x86,0x28,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0062886 <unknown>

// Aliases

mov     {z0.d, z1.d}, za.d[w8, 0, vgx2]  // 11000000-00000110-00001000-00000000
// CHECK-INST: mov     { z0.d, z1.d }, za.d[w8, 0, vgx2]
// CHECK-ENCODING: [0x00,0x08,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060800 <unknown>

mov     {z20.d, z21.d}, za.d[w10, 2, vgx2]  // 11000000-00000110-01001000-01010100
// CHECK-INST: mov     { z20.d, z21.d }, za.d[w10, 2, vgx2]
// CHECK-ENCODING: [0x54,0x48,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064854 <unknown>

mov     {z22.d, z23.d}, za.d[w11, 5, vgx2]  // 11000000-00000110-01101000-10110110
// CHECK-INST: mov     { z22.d, z23.d }, za.d[w11, 5, vgx2]
// CHECK-ENCODING: [0xb6,0x68,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00668b6 <unknown>

mov     {z30.d, z31.d}, za.d[w11, 7, vgx2]  // 11000000-00000110-01101000-11111110
// CHECK-INST: mov     { z30.d, z31.d }, za.d[w11, 7, vgx2]
// CHECK-ENCODING: [0xfe,0x68,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00668fe <unknown>

mov     {z4.d, z5.d}, za.d[w8, 1, vgx2]  // 11000000-00000110-00001000-00100100
// CHECK-INST: mov     { z4.d, z5.d }, za.d[w8, 1, vgx2]
// CHECK-ENCODING: [0x24,0x08,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060824 <unknown>

mov     {z0.d, z1.d}, za.d[w8, 1, vgx2]  // 11000000-00000110-00001000-00100000
// CHECK-INST: mov     { z0.d, z1.d }, za.d[w8, 1, vgx2]
// CHECK-ENCODING: [0x20,0x08,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060820 <unknown>

mov     {z24.d, z25.d}, za.d[w10, 3, vgx2]  // 11000000-00000110-01001000-01111000
// CHECK-INST: mov     { z24.d, z25.d }, za.d[w10, 3, vgx2]
// CHECK-ENCODING: [0x78,0x48,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064878 <unknown>

mov     {z0.d, z1.d}, za.d[w8, 4, vgx2]  // 11000000-00000110-00001000-10000000
// CHECK-INST: mov     { z0.d, z1.d }, za.d[w8, 4, vgx2]
// CHECK-ENCODING: [0x80,0x08,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060880 <unknown>

mov     {z16.d, z17.d}, za.d[w10, 1, vgx2]  // 11000000-00000110-01001000-00110000
// CHECK-INST: mov     { z16.d, z17.d }, za.d[w10, 1, vgx2]
// CHECK-ENCODING: [0x30,0x48,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064830 <unknown>

mov     {z28.d, z29.d}, za.d[w8, 6, vgx2]  // 11000000-00000110-00001000-11011100
// CHECK-INST: mov     { z28.d, z29.d }, za.d[w8, 6, vgx2]
// CHECK-ENCODING: [0xdc,0x08,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00608dc <unknown>

mov     {z2.d, z3.d}, za.d[w11, 1, vgx2]  // 11000000-00000110-01101000-00100010
// CHECK-INST: mov     { z2.d, z3.d }, za.d[w11, 1, vgx2]
// CHECK-ENCODING: [0x22,0x68,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066822 <unknown>

mov     {z6.d, z7.d}, za.d[w9, 4, vgx2]  // 11000000-00000110-00101000-10000110
// CHECK-INST: mov     { z6.d, z7.d }, za.d[w9, 4, vgx2]
// CHECK-ENCODING: [0x86,0x28,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0062886 <unknown>


mova    za0h.d[w12, 0:1], {z0.d, z1.d}  // 11000000-11000100-00000000-00000000
// CHECK-INST: mov     za0h.d[w12, 0:1], { z0.d, z1.d }
// CHECK-ENCODING: [0x00,0x00,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40000 <unknown>

mova    za5h.d[w14, 0:1], {z10.d, z11.d}  // 11000000-11000100-01000001-01000101
// CHECK-INST: mov     za5h.d[w14, 0:1], { z10.d, z11.d }
// CHECK-ENCODING: [0x45,0x41,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c44145 <unknown>

mova    za7h.d[w15, 0:1], {z12.d, z13.d}  // 11000000-11000100-01100001-10000111
// CHECK-INST: mov     za7h.d[w15, 0:1], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0x61,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c46187 <unknown>

mova    za7h.d[w15, 0:1], {z30.d, z31.d}  // 11000000-11000100-01100011-11000111
// CHECK-INST: mov     za7h.d[w15, 0:1], { z30.d, z31.d }
// CHECK-ENCODING: [0xc7,0x63,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c463c7 <unknown>

mova    za5h.d[w12, 0:1], {z16.d, z17.d}  // 11000000-11000100-00000010-00000101
// CHECK-INST: mov     za5h.d[w12, 0:1], { z16.d, z17.d }
// CHECK-ENCODING: [0x05,0x02,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40205 <unknown>

mova    za1h.d[w12, 0:1], {z0.d, z1.d}  // 11000000-11000100-00000000-00000001
// CHECK-INST: mov     za1h.d[w12, 0:1], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0x00,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40001 <unknown>

mova    za0h.d[w14, 0:1], {z18.d, z19.d}  // 11000000-11000100-01000010-01000000
// CHECK-INST: mov     za0h.d[w14, 0:1], { z18.d, z19.d }
// CHECK-ENCODING: [0x40,0x42,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c44240 <unknown>

mova    za0h.d[w12, 0:1], {z12.d, z13.d}  // 11000000-11000100-00000001-10000000
// CHECK-INST: mov     za0h.d[w12, 0:1], { z12.d, z13.d }
// CHECK-ENCODING: [0x80,0x01,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40180 <unknown>

mova    za1h.d[w14, 0:1], {z0.d, z1.d}  // 11000000-11000100-01000000-00000001
// CHECK-INST: mov     za1h.d[w14, 0:1], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0x40,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c44001 <unknown>

mova    za5h.d[w12, 0:1], {z22.d, z23.d}  // 11000000-11000100-00000010-11000101
// CHECK-INST: mov     za5h.d[w12, 0:1], { z22.d, z23.d }
// CHECK-ENCODING: [0xc5,0x02,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c402c5 <unknown>

mova    za2h.d[w15, 0:1], {z8.d, z9.d}  // 11000000-11000100-01100001-00000010
// CHECK-INST: mov     za2h.d[w15, 0:1], { z8.d, z9.d }
// CHECK-ENCODING: [0x02,0x61,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c46102 <unknown>

mova    za7h.d[w13, 0:1], {z12.d, z13.d}  // 11000000-11000100-00100001-10000111
// CHECK-INST: mov     za7h.d[w13, 0:1], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0x21,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c42187 <unknown>

// Aliases

mov     za0h.d[w12, 0:1], {z0.d, z1.d}  // 11000000-11000100-00000000-00000000
// CHECK-INST: mov     za0h.d[w12, 0:1], { z0.d, z1.d }
// CHECK-ENCODING: [0x00,0x00,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40000 <unknown>

mov     za5h.d[w14, 0:1], {z10.d, z11.d}  // 11000000-11000100-01000001-01000101
// CHECK-INST: mov     za5h.d[w14, 0:1], { z10.d, z11.d }
// CHECK-ENCODING: [0x45,0x41,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c44145 <unknown>

mov     za7h.d[w15, 0:1], {z12.d, z13.d}  // 11000000-11000100-01100001-10000111
// CHECK-INST: mov     za7h.d[w15, 0:1], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0x61,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c46187 <unknown>

mov     za7h.d[w15, 0:1], {z30.d, z31.d}  // 11000000-11000100-01100011-11000111
// CHECK-INST: mov     za7h.d[w15, 0:1], { z30.d, z31.d }
// CHECK-ENCODING: [0xc7,0x63,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c463c7 <unknown>

mov     za5h.d[w12, 0:1], {z16.d, z17.d}  // 11000000-11000100-00000010-00000101
// CHECK-INST: mov     za5h.d[w12, 0:1], { z16.d, z17.d }
// CHECK-ENCODING: [0x05,0x02,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40205 <unknown>

mov     za1h.d[w12, 0:1], {z0.d, z1.d}  // 11000000-11000100-00000000-00000001
// CHECK-INST: mov     za1h.d[w12, 0:1], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0x00,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40001 <unknown>

mov     za0h.d[w14, 0:1], {z18.d, z19.d}  // 11000000-11000100-01000010-01000000
// CHECK-INST: mov     za0h.d[w14, 0:1], { z18.d, z19.d }
// CHECK-ENCODING: [0x40,0x42,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c44240 <unknown>

mov     za0h.d[w12, 0:1], {z12.d, z13.d}  // 11000000-11000100-00000001-10000000
// CHECK-INST: mov     za0h.d[w12, 0:1], { z12.d, z13.d }
// CHECK-ENCODING: [0x80,0x01,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40180 <unknown>

mov     za1h.d[w14, 0:1], {z0.d, z1.d}  // 11000000-11000100-01000000-00000001
// CHECK-INST: mov     za1h.d[w14, 0:1], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0x40,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c44001 <unknown>

mov     za5h.d[w12, 0:1], {z22.d, z23.d}  // 11000000-11000100-00000010-11000101
// CHECK-INST: mov     za5h.d[w12, 0:1], { z22.d, z23.d }
// CHECK-ENCODING: [0xc5,0x02,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c402c5 <unknown>

mov     za2h.d[w15, 0:1], {z8.d, z9.d}  // 11000000-11000100-01100001-00000010
// CHECK-INST: mov     za2h.d[w15, 0:1], { z8.d, z9.d }
// CHECK-ENCODING: [0x02,0x61,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c46102 <unknown>

mov     za7h.d[w13, 0:1], {z12.d, z13.d}  // 11000000-11000100-00100001-10000111
// CHECK-INST: mov     za7h.d[w13, 0:1], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0x21,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c42187 <unknown>


mova    za0v.d[w12, 0:1], {z0.d, z1.d}  // 11000000-11000100-10000000-00000000
// CHECK-INST: mov     za0v.d[w12, 0:1], { z0.d, z1.d }
// CHECK-ENCODING: [0x00,0x80,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48000 <unknown>

mova    za5v.d[w14, 0:1], {z10.d, z11.d}  // 11000000-11000100-11000001-01000101
// CHECK-INST: mov     za5v.d[w14, 0:1], { z10.d, z11.d }
// CHECK-ENCODING: [0x45,0xc1,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4c145 <unknown>

mova    za7v.d[w15, 0:1], {z12.d, z13.d}  // 11000000-11000100-11100001-10000111
// CHECK-INST: mov     za7v.d[w15, 0:1], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0xe1,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4e187 <unknown>

mova    za7v.d[w15, 0:1], {z30.d, z31.d}  // 11000000-11000100-11100011-11000111
// CHECK-INST: mov     za7v.d[w15, 0:1], { z30.d, z31.d }
// CHECK-ENCODING: [0xc7,0xe3,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4e3c7 <unknown>

mova    za5v.d[w12, 0:1], {z16.d, z17.d}  // 11000000-11000100-10000010-00000101
// CHECK-INST: mov     za5v.d[w12, 0:1], { z16.d, z17.d }
// CHECK-ENCODING: [0x05,0x82,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48205 <unknown>

mova    za1v.d[w12, 0:1], {z0.d, z1.d}  // 11000000-11000100-10000000-00000001
// CHECK-INST: mov     za1v.d[w12, 0:1], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0x80,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48001 <unknown>

mova    za0v.d[w14, 0:1], {z18.d, z19.d}  // 11000000-11000100-11000010-01000000
// CHECK-INST: mov     za0v.d[w14, 0:1], { z18.d, z19.d }
// CHECK-ENCODING: [0x40,0xc2,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4c240 <unknown>

mova    za0v.d[w12, 0:1], {z12.d, z13.d}  // 11000000-11000100-10000001-10000000
// CHECK-INST: mov     za0v.d[w12, 0:1], { z12.d, z13.d }
// CHECK-ENCODING: [0x80,0x81,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48180 <unknown>

mova    za1v.d[w14, 0:1], {z0.d, z1.d}  // 11000000-11000100-11000000-00000001
// CHECK-INST: mov     za1v.d[w14, 0:1], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0xc0,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4c001 <unknown>

mova    za5v.d[w12, 0:1], {z22.d, z23.d}  // 11000000-11000100-10000010-11000101
// CHECK-INST: mov     za5v.d[w12, 0:1], { z22.d, z23.d }
// CHECK-ENCODING: [0xc5,0x82,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c482c5 <unknown>

mova    za2v.d[w15, 0:1], {z8.d, z9.d}  // 11000000-11000100-11100001-00000010
// CHECK-INST: mov     za2v.d[w15, 0:1], { z8.d, z9.d }
// CHECK-ENCODING: [0x02,0xe1,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4e102 <unknown>

mova    za7v.d[w13, 0:1], {z12.d, z13.d}  // 11000000-11000100-10100001-10000111
// CHECK-INST: mov     za7v.d[w13, 0:1], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0xa1,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4a187 <unknown>

// Aliases

mov     za0v.d[w12, 0:1], {z0.d, z1.d}  // 11000000-11000100-10000000-00000000
// CHECK-INST: mov     za0v.d[w12, 0:1], { z0.d, z1.d }
// CHECK-ENCODING: [0x00,0x80,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48000 <unknown>

mov     za5v.d[w14, 0:1], {z10.d, z11.d}  // 11000000-11000100-11000001-01000101
// CHECK-INST: mov     za5v.d[w14, 0:1], { z10.d, z11.d }
// CHECK-ENCODING: [0x45,0xc1,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4c145 <unknown>

mov     za7v.d[w15, 0:1], {z12.d, z13.d}  // 11000000-11000100-11100001-10000111
// CHECK-INST: mov     za7v.d[w15, 0:1], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0xe1,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4e187 <unknown>

mov     za7v.d[w15, 0:1], {z30.d, z31.d}  // 11000000-11000100-11100011-11000111
// CHECK-INST: mov     za7v.d[w15, 0:1], { z30.d, z31.d }
// CHECK-ENCODING: [0xc7,0xe3,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4e3c7 <unknown>

mov     za5v.d[w12, 0:1], {z16.d, z17.d}  // 11000000-11000100-10000010-00000101
// CHECK-INST: mov     za5v.d[w12, 0:1], { z16.d, z17.d }
// CHECK-ENCODING: [0x05,0x82,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48205 <unknown>

mov     za1v.d[w12, 0:1], {z0.d, z1.d}  // 11000000-11000100-10000000-00000001
// CHECK-INST: mov     za1v.d[w12, 0:1], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0x80,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48001 <unknown>

mov     za0v.d[w14, 0:1], {z18.d, z19.d}  // 11000000-11000100-11000010-01000000
// CHECK-INST: mov     za0v.d[w14, 0:1], { z18.d, z19.d }
// CHECK-ENCODING: [0x40,0xc2,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4c240 <unknown>

mov     za0v.d[w12, 0:1], {z12.d, z13.d}  // 11000000-11000100-10000001-10000000
// CHECK-INST: mov     za0v.d[w12, 0:1], { z12.d, z13.d }
// CHECK-ENCODING: [0x80,0x81,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48180 <unknown>

mov     za1v.d[w14, 0:1], {z0.d, z1.d}  // 11000000-11000100-11000000-00000001
// CHECK-INST: mov     za1v.d[w14, 0:1], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0xc0,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4c001 <unknown>

mov     za5v.d[w12, 0:1], {z22.d, z23.d}  // 11000000-11000100-10000010-11000101
// CHECK-INST: mov     za5v.d[w12, 0:1], { z22.d, z23.d }
// CHECK-ENCODING: [0xc5,0x82,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c482c5 <unknown>

mov     za2v.d[w15, 0:1], {z8.d, z9.d}  // 11000000-11000100-11100001-00000010
// CHECK-INST: mov     za2v.d[w15, 0:1], { z8.d, z9.d }
// CHECK-ENCODING: [0x02,0xe1,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4e102 <unknown>

mov     za7v.d[w13, 0:1], {z12.d, z13.d}  // 11000000-11000100-10100001-10000111
// CHECK-INST: mov     za7v.d[w13, 0:1], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0xa1,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4a187 <unknown>


mova    za.d[w8, 0, vgx2], {z0.d, z1.d}  // 11000000-00000100-00001000-00000000
// CHECK-INST: mov     za.d[w8, 0, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x00,0x08,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040800 <unknown>

mova    za.d[w8, 0], {z0.d, z1.d}  // 11000000-00000100-00001000-00000000
// CHECK-INST: mov     za.d[w8, 0, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x00,0x08,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040800 <unknown>

mova    za.d[w10, 5, vgx2], {z10.d, z11.d}  // 11000000-00000100-01001001-01000101
// CHECK-INST: mov     za.d[w10, 5, vgx2], { z10.d, z11.d }
// CHECK-ENCODING: [0x45,0x49,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044945 <unknown>

mova    za.d[w10, 5], {z10.d, z11.d}  // 11000000-00000100-01001001-01000101
// CHECK-INST: mov     za.d[w10, 5, vgx2], { z10.d, z11.d }
// CHECK-ENCODING: [0x45,0x49,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044945 <unknown>

mova    za.d[w11, 7, vgx2], {z12.d, z13.d}  // 11000000-00000100-01101001-10000111
// CHECK-INST: mov     za.d[w11, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0x69,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046987 <unknown>

mova    za.d[w11, 7], {z12.d, z13.d}  // 11000000-00000100-01101001-10000111
// CHECK-INST: mov     za.d[w11, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0x69,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046987 <unknown>

mova    za.d[w11, 7, vgx2], {z30.d, z31.d}  // 11000000-00000100-01101011-11000111
// CHECK-INST: mov     za.d[w11, 7, vgx2], { z30.d, z31.d }
// CHECK-ENCODING: [0xc7,0x6b,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046bc7 <unknown>

mova    za.d[w11, 7], {z30.d, z31.d}  // 11000000-00000100-01101011-11000111
// CHECK-INST: mov     za.d[w11, 7, vgx2], { z30.d, z31.d }
// CHECK-ENCODING: [0xc7,0x6b,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046bc7 <unknown>

mova    za.d[w8, 5, vgx2], {z16.d, z17.d}  // 11000000-00000100-00001010-00000101
// CHECK-INST: mov     za.d[w8, 5, vgx2], { z16.d, z17.d }
// CHECK-ENCODING: [0x05,0x0a,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040a05 <unknown>

mova    za.d[w8, 5], {z16.d, z17.d}  // 11000000-00000100-00001010-00000101
// CHECK-INST: mov     za.d[w8, 5, vgx2], { z16.d, z17.d }
// CHECK-ENCODING: [0x05,0x0a,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040a05 <unknown>

mova    za.d[w8, 1, vgx2], {z0.d, z1.d}  // 11000000-00000100-00001000-00000001
// CHECK-INST: mov     za.d[w8, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0x08,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040801 <unknown>

mova    za.d[w8, 1], {z0.d, z1.d}  // 11000000-00000100-00001000-00000001
// CHECK-INST: mov     za.d[w8, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0x08,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040801 <unknown>

mova    za.d[w10, 0, vgx2], {z18.d, z19.d}  // 11000000-00000100-01001010-01000000
// CHECK-INST: mov     za.d[w10, 0, vgx2], { z18.d, z19.d }
// CHECK-ENCODING: [0x40,0x4a,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044a40 <unknown>

mova    za.d[w10, 0], {z18.d, z19.d}  // 11000000-00000100-01001010-01000000
// CHECK-INST: mov     za.d[w10, 0, vgx2], { z18.d, z19.d }
// CHECK-ENCODING: [0x40,0x4a,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044a40 <unknown>

mova    za.d[w8, 0, vgx2], {z12.d, z13.d}  // 11000000-00000100-00001001-10000000
// CHECK-INST: mov     za.d[w8, 0, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x80,0x09,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040980 <unknown>

mova    za.d[w8, 0], {z12.d, z13.d}  // 11000000-00000100-00001001-10000000
// CHECK-INST: mov     za.d[w8, 0, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x80,0x09,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040980 <unknown>

mova    za.d[w10, 1, vgx2], {z0.d, z1.d}  // 11000000-00000100-01001000-00000001
// CHECK-INST: mov     za.d[w10, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0x48,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044801 <unknown>

mova    za.d[w10, 1], {z0.d, z1.d}  // 11000000-00000100-01001000-00000001
// CHECK-INST: mov     za.d[w10, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0x48,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044801 <unknown>

mova    za.d[w8, 5, vgx2], {z22.d, z23.d}  // 11000000-00000100-00001010-11000101
// CHECK-INST: mov     za.d[w8, 5, vgx2], { z22.d, z23.d }
// CHECK-ENCODING: [0xc5,0x0a,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040ac5 <unknown>

mova    za.d[w8, 5], {z22.d, z23.d}  // 11000000-00000100-00001010-11000101
// CHECK-INST: mov     za.d[w8, 5, vgx2], { z22.d, z23.d }
// CHECK-ENCODING: [0xc5,0x0a,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040ac5 <unknown>

mova    za.d[w11, 2, vgx2], {z8.d, z9.d}  // 11000000-00000100-01101001-00000010
// CHECK-INST: mov     za.d[w11, 2, vgx2], { z8.d, z9.d }
// CHECK-ENCODING: [0x02,0x69,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046902 <unknown>

mova    za.d[w11, 2], {z8.d, z9.d}  // 11000000-00000100-01101001-00000010
// CHECK-INST: mov     za.d[w11, 2, vgx2], { z8.d, z9.d }
// CHECK-ENCODING: [0x02,0x69,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046902 <unknown>

mova    za.d[w9, 7, vgx2], {z12.d, z13.d}  // 11000000-00000100-00101001-10000111
// CHECK-INST: mov     za.d[w9, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0x29,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0042987 <unknown>

mova    za.d[w9, 7], {z12.d, z13.d}  // 11000000-00000100-00101001-10000111
// CHECK-INST: mov     za.d[w9, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0x29,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0042987 <unknown>

// Aliases

mov     za.d[w8, 0, vgx2], {z0.d, z1.d}  // 11000000-00000100-00001000-00000000
// CHECK-INST: mov     za.d[w8, 0, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x00,0x08,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040800 <unknown>

mov     za.d[w10, 5, vgx2], {z10.d, z11.d}  // 11000000-00000100-01001001-01000101
// CHECK-INST: mov     za.d[w10, 5, vgx2], { z10.d, z11.d }
// CHECK-ENCODING: [0x45,0x49,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044945 <unknown>

mov     za.d[w11, 7, vgx2], {z12.d, z13.d}  // 11000000-00000100-01101001-10000111
// CHECK-INST: mov     za.d[w11, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0x69,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046987 <unknown>

mov     za.d[w11, 7, vgx2], {z30.d, z31.d}  // 11000000-00000100-01101011-11000111
// CHECK-INST: mov     za.d[w11, 7, vgx2], { z30.d, z31.d }
// CHECK-ENCODING: [0xc7,0x6b,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046bc7 <unknown>

mov     za.d[w8, 5, vgx2], {z16.d, z17.d}  // 11000000-00000100-00001010-00000101
// CHECK-INST: mov     za.d[w8, 5, vgx2], { z16.d, z17.d }
// CHECK-ENCODING: [0x05,0x0a,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040a05 <unknown>

mov     za.d[w8, 1, vgx2], {z0.d, z1.d}  // 11000000-00000100-00001000-00000001
// CHECK-INST: mov     za.d[w8, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0x08,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040801 <unknown>

mov     za.d[w10, 0, vgx2], {z18.d, z19.d}  // 11000000-00000100-01001010-01000000
// CHECK-INST: mov     za.d[w10, 0, vgx2], { z18.d, z19.d }
// CHECK-ENCODING: [0x40,0x4a,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044a40 <unknown>

mov     za.d[w8, 0, vgx2], {z12.d, z13.d}  // 11000000-00000100-00001001-10000000
// CHECK-INST: mov     za.d[w8, 0, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x80,0x09,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040980 <unknown>

mov     za.d[w10, 1, vgx2], {z0.d, z1.d}  // 11000000-00000100-01001000-00000001
// CHECK-INST: mov     za.d[w10, 1, vgx2], { z0.d, z1.d }
// CHECK-ENCODING: [0x01,0x48,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044801 <unknown>

mov     za.d[w8, 5, vgx2], {z22.d, z23.d}  // 11000000-00000100-00001010-11000101
// CHECK-INST: mov     za.d[w8, 5, vgx2], { z22.d, z23.d }
// CHECK-ENCODING: [0xc5,0x0a,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040ac5 <unknown>

mov     za.d[w11, 2, vgx2], {z8.d, z9.d}  // 11000000-00000100-01101001-00000010
// CHECK-INST: mov     za.d[w11, 2, vgx2], { z8.d, z9.d }
// CHECK-ENCODING: [0x02,0x69,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046902 <unknown>

mov     za.d[w9, 7, vgx2], {z12.d, z13.d}  // 11000000-00000100-00101001-10000111
// CHECK-INST: mov     za.d[w9, 7, vgx2], { z12.d, z13.d }
// CHECK-ENCODING: [0x87,0x29,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0042987 <unknown>


mova    {z0.b, z1.b}, za0h.b[w12, 0:1]  // 11000000-00000110-00000000-00000000
// CHECK-INST: mov     { z0.b, z1.b }, za0h.b[w12, 0:1]
// CHECK-ENCODING: [0x00,0x00,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060000 <unknown>

mova    {z20.b, z21.b}, za0h.b[w14, 4:5]  // 11000000-00000110-01000000-01010100
// CHECK-INST: mov     { z20.b, z21.b }, za0h.b[w14, 4:5]
// CHECK-ENCODING: [0x54,0x40,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064054 <unknown>

mova    {z22.b, z23.b}, za0h.b[w15, 10:11]  // 11000000-00000110-01100000-10110110
// CHECK-INST: mov     { z22.b, z23.b }, za0h.b[w15, 10:11]
// CHECK-ENCODING: [0xb6,0x60,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00660b6 <unknown>

mova    {z30.b, z31.b}, za0h.b[w15, 14:15]  // 11000000-00000110-01100000-11111110
// CHECK-INST: mov     { z30.b, z31.b }, za0h.b[w15, 14:15]
// CHECK-ENCODING: [0xfe,0x60,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00660fe <unknown>

mova    {z4.b, z5.b}, za0h.b[w12, 2:3]  // 11000000-00000110-00000000-00100100
// CHECK-INST: mov     { z4.b, z5.b }, za0h.b[w12, 2:3]
// CHECK-ENCODING: [0x24,0x00,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060024 <unknown>

mova    {z0.b, z1.b}, za0h.b[w12, 2:3]  // 11000000-00000110-00000000-00100000
// CHECK-INST: mov     { z0.b, z1.b }, za0h.b[w12, 2:3]
// CHECK-ENCODING: [0x20,0x00,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060020 <unknown>

mova    {z24.b, z25.b}, za0h.b[w14, 6:7]  // 11000000-00000110-01000000-01111000
// CHECK-INST: mov     { z24.b, z25.b }, za0h.b[w14, 6:7]
// CHECK-ENCODING: [0x78,0x40,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064078 <unknown>

mova    {z0.b, z1.b}, za0h.b[w12, 8:9]  // 11000000-00000110-00000000-10000000
// CHECK-INST: mov     { z0.b, z1.b }, za0h.b[w12, 8:9]
// CHECK-ENCODING: [0x80,0x00,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060080 <unknown>

mova    {z16.b, z17.b}, za0h.b[w14, 2:3]  // 11000000-00000110-01000000-00110000
// CHECK-INST: mov     { z16.b, z17.b }, za0h.b[w14, 2:3]
// CHECK-ENCODING: [0x30,0x40,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064030 <unknown>

mova    {z28.b, z29.b}, za0h.b[w12, 12:13]  // 11000000-00000110-00000000-11011100
// CHECK-INST: mov     { z28.b, z29.b }, za0h.b[w12, 12:13]
// CHECK-ENCODING: [0xdc,0x00,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00600dc <unknown>

mova    {z2.b, z3.b}, za0h.b[w15, 2:3]  // 11000000-00000110-01100000-00100010
// CHECK-INST: mov     { z2.b, z3.b }, za0h.b[w15, 2:3]
// CHECK-ENCODING: [0x22,0x60,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066022 <unknown>

mova    {z6.b, z7.b}, za0h.b[w13, 8:9]  // 11000000-00000110-00100000-10000110
// CHECK-INST: mov     { z6.b, z7.b }, za0h.b[w13, 8:9]
// CHECK-ENCODING: [0x86,0x20,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0062086 <unknown>

// Aliases

mov     {z0.b, z1.b}, za0h.b[w12, 0:1]  // 11000000-00000110-00000000-00000000
// CHECK-INST: mov     { z0.b, z1.b }, za0h.b[w12, 0:1]
// CHECK-ENCODING: [0x00,0x00,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060000 <unknown>

mov     {z20.b, z21.b}, za0h.b[w14, 4:5]  // 11000000-00000110-01000000-01010100
// CHECK-INST: mov     { z20.b, z21.b }, za0h.b[w14, 4:5]
// CHECK-ENCODING: [0x54,0x40,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064054 <unknown>

mov     {z22.b, z23.b}, za0h.b[w15, 10:11]  // 11000000-00000110-01100000-10110110
// CHECK-INST: mov     { z22.b, z23.b }, za0h.b[w15, 10:11]
// CHECK-ENCODING: [0xb6,0x60,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00660b6 <unknown>

mov     {z30.b, z31.b}, za0h.b[w15, 14:15]  // 11000000-00000110-01100000-11111110
// CHECK-INST: mov     { z30.b, z31.b }, za0h.b[w15, 14:15]
// CHECK-ENCODING: [0xfe,0x60,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00660fe <unknown>

mov     {z4.b, z5.b}, za0h.b[w12, 2:3]  // 11000000-00000110-00000000-00100100
// CHECK-INST: mov     { z4.b, z5.b }, za0h.b[w12, 2:3]
// CHECK-ENCODING: [0x24,0x00,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060024 <unknown>

mov     {z0.b, z1.b}, za0h.b[w12, 2:3]  // 11000000-00000110-00000000-00100000
// CHECK-INST: mov     { z0.b, z1.b }, za0h.b[w12, 2:3]
// CHECK-ENCODING: [0x20,0x00,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060020 <unknown>

mov     {z24.b, z25.b}, za0h.b[w14, 6:7]  // 11000000-00000110-01000000-01111000
// CHECK-INST: mov     { z24.b, z25.b }, za0h.b[w14, 6:7]
// CHECK-ENCODING: [0x78,0x40,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064078 <unknown>

mov     {z0.b, z1.b}, za0h.b[w12, 8:9]  // 11000000-00000110-00000000-10000000
// CHECK-INST: mov     { z0.b, z1.b }, za0h.b[w12, 8:9]
// CHECK-ENCODING: [0x80,0x00,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060080 <unknown>

mov     {z16.b, z17.b}, za0h.b[w14, 2:3]  // 11000000-00000110-01000000-00110000
// CHECK-INST: mov     { z16.b, z17.b }, za0h.b[w14, 2:3]
// CHECK-ENCODING: [0x30,0x40,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064030 <unknown>

mov     {z28.b, z29.b}, za0h.b[w12, 12:13]  // 11000000-00000110-00000000-11011100
// CHECK-INST: mov     { z28.b, z29.b }, za0h.b[w12, 12:13]
// CHECK-ENCODING: [0xdc,0x00,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00600dc <unknown>

mov     {z2.b, z3.b}, za0h.b[w15, 2:3]  // 11000000-00000110-01100000-00100010
// CHECK-INST: mov     { z2.b, z3.b }, za0h.b[w15, 2:3]
// CHECK-ENCODING: [0x22,0x60,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066022 <unknown>

mov     {z6.b, z7.b}, za0h.b[w13, 8:9]  // 11000000-00000110-00100000-10000110
// CHECK-INST: mov     { z6.b, z7.b }, za0h.b[w13, 8:9]
// CHECK-ENCODING: [0x86,0x20,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0062086 <unknown>


mova    {z0.b, z1.b}, za0v.b[w12, 0:1]  // 11000000-00000110-10000000-00000000
// CHECK-INST: mov     { z0.b, z1.b }, za0v.b[w12, 0:1]
// CHECK-ENCODING: [0x00,0x80,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0068000 <unknown>

mova    {z20.b, z21.b}, za0v.b[w14, 4:5]  // 11000000-00000110-11000000-01010100
// CHECK-INST: mov     { z20.b, z21.b }, za0v.b[w14, 4:5]
// CHECK-ENCODING: [0x54,0xc0,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006c054 <unknown>

mova    {z22.b, z23.b}, za0v.b[w15, 10:11]  // 11000000-00000110-11100000-10110110
// CHECK-INST: mov     { z22.b, z23.b }, za0v.b[w15, 10:11]
// CHECK-ENCODING: [0xb6,0xe0,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006e0b6 <unknown>

mova    {z30.b, z31.b}, za0v.b[w15, 14:15]  // 11000000-00000110-11100000-11111110
// CHECK-INST: mov     { z30.b, z31.b }, za0v.b[w15, 14:15]
// CHECK-ENCODING: [0xfe,0xe0,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006e0fe <unknown>

mova    {z4.b, z5.b}, za0v.b[w12, 2:3]  // 11000000-00000110-10000000-00100100
// CHECK-INST: mov     { z4.b, z5.b }, za0v.b[w12, 2:3]
// CHECK-ENCODING: [0x24,0x80,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0068024 <unknown>

mova    {z0.b, z1.b}, za0v.b[w12, 2:3]  // 11000000-00000110-10000000-00100000
// CHECK-INST: mov     { z0.b, z1.b }, za0v.b[w12, 2:3]
// CHECK-ENCODING: [0x20,0x80,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0068020 <unknown>

mova    {z24.b, z25.b}, za0v.b[w14, 6:7]  // 11000000-00000110-11000000-01111000
// CHECK-INST: mov     { z24.b, z25.b }, za0v.b[w14, 6:7]
// CHECK-ENCODING: [0x78,0xc0,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006c078 <unknown>

mova    {z0.b, z1.b}, za0v.b[w12, 8:9]  // 11000000-00000110-10000000-10000000
// CHECK-INST: mov     { z0.b, z1.b }, za0v.b[w12, 8:9]
// CHECK-ENCODING: [0x80,0x80,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0068080 <unknown>

mova    {z16.b, z17.b}, za0v.b[w14, 2:3]  // 11000000-00000110-11000000-00110000
// CHECK-INST: mov     { z16.b, z17.b }, za0v.b[w14, 2:3]
// CHECK-ENCODING: [0x30,0xc0,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006c030 <unknown>

mova    {z28.b, z29.b}, za0v.b[w12, 12:13]  // 11000000-00000110-10000000-11011100
// CHECK-INST: mov     { z28.b, z29.b }, za0v.b[w12, 12:13]
// CHECK-ENCODING: [0xdc,0x80,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00680dc <unknown>

mova    {z2.b, z3.b}, za0v.b[w15, 2:3]  // 11000000-00000110-11100000-00100010
// CHECK-INST: mov     { z2.b, z3.b }, za0v.b[w15, 2:3]
// CHECK-ENCODING: [0x22,0xe0,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006e022 <unknown>

mova    {z6.b, z7.b}, za0v.b[w13, 8:9]  // 11000000-00000110-10100000-10000110
// CHECK-INST: mov     { z6.b, z7.b }, za0v.b[w13, 8:9]
// CHECK-ENCODING: [0x86,0xa0,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006a086 <unknown>

// Aliases

mov     {z0.b, z1.b}, za0v.b[w12, 0:1]  // 11000000-00000110-10000000-00000000
// CHECK-INST: mov     { z0.b, z1.b }, za0v.b[w12, 0:1]
// CHECK-ENCODING: [0x00,0x80,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0068000 <unknown>

mov     {z20.b, z21.b}, za0v.b[w14, 4:5]  // 11000000-00000110-11000000-01010100
// CHECK-INST: mov     { z20.b, z21.b }, za0v.b[w14, 4:5]
// CHECK-ENCODING: [0x54,0xc0,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006c054 <unknown>

mov     {z22.b, z23.b}, za0v.b[w15, 10:11]  // 11000000-00000110-11100000-10110110
// CHECK-INST: mov     { z22.b, z23.b }, za0v.b[w15, 10:11]
// CHECK-ENCODING: [0xb6,0xe0,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006e0b6 <unknown>

mov     {z30.b, z31.b}, za0v.b[w15, 14:15]  // 11000000-00000110-11100000-11111110
// CHECK-INST: mov     { z30.b, z31.b }, za0v.b[w15, 14:15]
// CHECK-ENCODING: [0xfe,0xe0,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006e0fe <unknown>

mov     {z4.b, z5.b}, za0v.b[w12, 2:3]  // 11000000-00000110-10000000-00100100
// CHECK-INST: mov     { z4.b, z5.b }, za0v.b[w12, 2:3]
// CHECK-ENCODING: [0x24,0x80,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0068024 <unknown>

mov     {z0.b, z1.b}, za0v.b[w12, 2:3]  // 11000000-00000110-10000000-00100000
// CHECK-INST: mov     { z0.b, z1.b }, za0v.b[w12, 2:3]
// CHECK-ENCODING: [0x20,0x80,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0068020 <unknown>

mov     {z24.b, z25.b}, za0v.b[w14, 6:7]  // 11000000-00000110-11000000-01111000
// CHECK-INST: mov     { z24.b, z25.b }, za0v.b[w14, 6:7]
// CHECK-ENCODING: [0x78,0xc0,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006c078 <unknown>

mov     {z0.b, z1.b}, za0v.b[w12, 8:9]  // 11000000-00000110-10000000-10000000
// CHECK-INST: mov     { z0.b, z1.b }, za0v.b[w12, 8:9]
// CHECK-ENCODING: [0x80,0x80,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0068080 <unknown>

mov     {z16.b, z17.b}, za0v.b[w14, 2:3]  // 11000000-00000110-11000000-00110000
// CHECK-INST: mov     { z16.b, z17.b }, za0v.b[w14, 2:3]
// CHECK-ENCODING: [0x30,0xc0,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006c030 <unknown>

mov     {z28.b, z29.b}, za0v.b[w12, 12:13]  // 11000000-00000110-10000000-11011100
// CHECK-INST: mov     { z28.b, z29.b }, za0v.b[w12, 12:13]
// CHECK-ENCODING: [0xdc,0x80,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00680dc <unknown>

mov     {z2.b, z3.b}, za0v.b[w15, 2:3]  // 11000000-00000110-11100000-00100010
// CHECK-INST: mov     { z2.b, z3.b }, za0v.b[w15, 2:3]
// CHECK-ENCODING: [0x22,0xe0,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006e022 <unknown>

mov     {z6.b, z7.b}, za0v.b[w13, 8:9]  // 11000000-00000110-10100000-10000110
// CHECK-INST: mov     { z6.b, z7.b }, za0v.b[w13, 8:9]
// CHECK-ENCODING: [0x86,0xa0,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006a086 <unknown>


mova    za0h.b[w12, 0:1], {z0.b, z1.b}  // 11000000-00000100-00000000-00000000
// CHECK-INST: mov     za0h.b[w12, 0:1], { z0.b, z1.b }
// CHECK-ENCODING: [0x00,0x00,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040000 <unknown>

mova    za0h.b[w14, 10:11], {z10.b, z11.b}  // 11000000-00000100-01000001-01000101
// CHECK-INST: mov     za0h.b[w14, 10:11], { z10.b, z11.b }
// CHECK-ENCODING: [0x45,0x41,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044145 <unknown>

mova    za0h.b[w15, 14:15], {z12.b, z13.b}  // 11000000-00000100-01100001-10000111
// CHECK-INST: mov     za0h.b[w15, 14:15], { z12.b, z13.b }
// CHECK-ENCODING: [0x87,0x61,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046187 <unknown>

mova    za0h.b[w15, 14:15], {z30.b, z31.b}  // 11000000-00000100-01100011-11000111
// CHECK-INST: mov     za0h.b[w15, 14:15], { z30.b, z31.b }
// CHECK-ENCODING: [0xc7,0x63,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00463c7 <unknown>

mova    za0h.b[w12, 10:11], {z16.b, z17.b}  // 11000000-00000100-00000010-00000101
// CHECK-INST: mov     za0h.b[w12, 10:11], { z16.b, z17.b }
// CHECK-ENCODING: [0x05,0x02,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040205 <unknown>

mova    za0h.b[w12, 2:3], {z0.b, z1.b}  // 11000000-00000100-00000000-00000001
// CHECK-INST: mov     za0h.b[w12, 2:3], { z0.b, z1.b }
// CHECK-ENCODING: [0x01,0x00,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040001 <unknown>

mova    za0h.b[w14, 0:1], {z18.b, z19.b}  // 11000000-00000100-01000010-01000000
// CHECK-INST: mov     za0h.b[w14, 0:1], { z18.b, z19.b }
// CHECK-ENCODING: [0x40,0x42,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044240 <unknown>

mova    za0h.b[w12, 0:1], {z12.b, z13.b}  // 11000000-00000100-00000001-10000000
// CHECK-INST: mov     za0h.b[w12, 0:1], { z12.b, z13.b }
// CHECK-ENCODING: [0x80,0x01,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040180 <unknown>

mova    za0h.b[w14, 2:3], {z0.b, z1.b}  // 11000000-00000100-01000000-00000001
// CHECK-INST: mov     za0h.b[w14, 2:3], { z0.b, z1.b }
// CHECK-ENCODING: [0x01,0x40,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044001 <unknown>

mova    za0h.b[w12, 10:11], {z22.b, z23.b}  // 11000000-00000100-00000010-11000101
// CHECK-INST: mov     za0h.b[w12, 10:11], { z22.b, z23.b }
// CHECK-ENCODING: [0xc5,0x02,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00402c5 <unknown>

mova    za0h.b[w15, 4:5], {z8.b, z9.b}  // 11000000-00000100-01100001-00000010
// CHECK-INST: mov     za0h.b[w15, 4:5], { z8.b, z9.b }
// CHECK-ENCODING: [0x02,0x61,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046102 <unknown>

mova    za0h.b[w13, 14:15], {z12.b, z13.b}  // 11000000-00000100-00100001-10000111
// CHECK-INST: mov     za0h.b[w13, 14:15], { z12.b, z13.b }
// CHECK-ENCODING: [0x87,0x21,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0042187 <unknown>

// Aliases

mov     za0h.b[w12, 0:1], {z0.b, z1.b}  // 11000000-00000100-00000000-00000000
// CHECK-INST: mov     za0h.b[w12, 0:1], { z0.b, z1.b }
// CHECK-ENCODING: [0x00,0x00,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040000 <unknown>

mov     za0h.b[w14, 10:11], {z10.b, z11.b}  // 11000000-00000100-01000001-01000101
// CHECK-INST: mov     za0h.b[w14, 10:11], { z10.b, z11.b }
// CHECK-ENCODING: [0x45,0x41,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044145 <unknown>

mov     za0h.b[w15, 14:15], {z12.b, z13.b}  // 11000000-00000100-01100001-10000111
// CHECK-INST: mov     za0h.b[w15, 14:15], { z12.b, z13.b }
// CHECK-ENCODING: [0x87,0x61,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046187 <unknown>

mov     za0h.b[w15, 14:15], {z30.b, z31.b}  // 11000000-00000100-01100011-11000111
// CHECK-INST: mov     za0h.b[w15, 14:15], { z30.b, z31.b }
// CHECK-ENCODING: [0xc7,0x63,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00463c7 <unknown>

mov     za0h.b[w12, 10:11], {z16.b, z17.b}  // 11000000-00000100-00000010-00000101
// CHECK-INST: mov     za0h.b[w12, 10:11], { z16.b, z17.b }
// CHECK-ENCODING: [0x05,0x02,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040205 <unknown>

mov     za0h.b[w12, 2:3], {z0.b, z1.b}  // 11000000-00000100-00000000-00000001
// CHECK-INST: mov     za0h.b[w12, 2:3], { z0.b, z1.b }
// CHECK-ENCODING: [0x01,0x00,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040001 <unknown>

mov     za0h.b[w14, 0:1], {z18.b, z19.b}  // 11000000-00000100-01000010-01000000
// CHECK-INST: mov     za0h.b[w14, 0:1], { z18.b, z19.b }
// CHECK-ENCODING: [0x40,0x42,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044240 <unknown>

mov     za0h.b[w12, 0:1], {z12.b, z13.b}  // 11000000-00000100-00000001-10000000
// CHECK-INST: mov     za0h.b[w12, 0:1], { z12.b, z13.b }
// CHECK-ENCODING: [0x80,0x01,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040180 <unknown>

mov     za0h.b[w14, 2:3], {z0.b, z1.b}  // 11000000-00000100-01000000-00000001
// CHECK-INST: mov     za0h.b[w14, 2:3], { z0.b, z1.b }
// CHECK-ENCODING: [0x01,0x40,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044001 <unknown>

mov     za0h.b[w12, 10:11], {z22.b, z23.b}  // 11000000-00000100-00000010-11000101
// CHECK-INST: mov     za0h.b[w12, 10:11], { z22.b, z23.b }
// CHECK-ENCODING: [0xc5,0x02,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00402c5 <unknown>

mov     za0h.b[w15, 4:5], {z8.b, z9.b}  // 11000000-00000100-01100001-00000010
// CHECK-INST: mov     za0h.b[w15, 4:5], { z8.b, z9.b }
// CHECK-ENCODING: [0x02,0x61,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046102 <unknown>

mov     za0h.b[w13, 14:15], {z12.b, z13.b}  // 11000000-00000100-00100001-10000111
// CHECK-INST: mov     za0h.b[w13, 14:15], { z12.b, z13.b }
// CHECK-ENCODING: [0x87,0x21,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0042187 <unknown>


mova    za0v.b[w12, 0:1], {z0.b, z1.b}  // 11000000-00000100-10000000-00000000
// CHECK-INST: mov     za0v.b[w12, 0:1], { z0.b, z1.b }
// CHECK-ENCODING: [0x00,0x80,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048000 <unknown>

mova    za0v.b[w14, 10:11], {z10.b, z11.b}  // 11000000-00000100-11000001-01000101
// CHECK-INST: mov     za0v.b[w14, 10:11], { z10.b, z11.b }
// CHECK-ENCODING: [0x45,0xc1,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004c145 <unknown>

mova    za0v.b[w15, 14:15], {z12.b, z13.b}  // 11000000-00000100-11100001-10000111
// CHECK-INST: mov     za0v.b[w15, 14:15], { z12.b, z13.b }
// CHECK-ENCODING: [0x87,0xe1,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004e187 <unknown>

mova    za0v.b[w15, 14:15], {z30.b, z31.b}  // 11000000-00000100-11100011-11000111
// CHECK-INST: mov     za0v.b[w15, 14:15], { z30.b, z31.b }
// CHECK-ENCODING: [0xc7,0xe3,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004e3c7 <unknown>

mova    za0v.b[w12, 10:11], {z16.b, z17.b}  // 11000000-00000100-10000010-00000101
// CHECK-INST: mov     za0v.b[w12, 10:11], { z16.b, z17.b }
// CHECK-ENCODING: [0x05,0x82,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048205 <unknown>

mova    za0v.b[w12, 2:3], {z0.b, z1.b}  // 11000000-00000100-10000000-00000001
// CHECK-INST: mov     za0v.b[w12, 2:3], { z0.b, z1.b }
// CHECK-ENCODING: [0x01,0x80,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048001 <unknown>

mova    za0v.b[w14, 0:1], {z18.b, z19.b}  // 11000000-00000100-11000010-01000000
// CHECK-INST: mov     za0v.b[w14, 0:1], { z18.b, z19.b }
// CHECK-ENCODING: [0x40,0xc2,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004c240 <unknown>

mova    za0v.b[w12, 0:1], {z12.b, z13.b}  // 11000000-00000100-10000001-10000000
// CHECK-INST: mov     za0v.b[w12, 0:1], { z12.b, z13.b }
// CHECK-ENCODING: [0x80,0x81,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048180 <unknown>

mova    za0v.b[w14, 2:3], {z0.b, z1.b}  // 11000000-00000100-11000000-00000001
// CHECK-INST: mov     za0v.b[w14, 2:3], { z0.b, z1.b }
// CHECK-ENCODING: [0x01,0xc0,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004c001 <unknown>

mova    za0v.b[w12, 10:11], {z22.b, z23.b}  // 11000000-00000100-10000010-11000101
// CHECK-INST: mov     za0v.b[w12, 10:11], { z22.b, z23.b }
// CHECK-ENCODING: [0xc5,0x82,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00482c5 <unknown>

mova    za0v.b[w15, 4:5], {z8.b, z9.b}  // 11000000-00000100-11100001-00000010
// CHECK-INST: mov     za0v.b[w15, 4:5], { z8.b, z9.b }
// CHECK-ENCODING: [0x02,0xe1,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004e102 <unknown>

mova    za0v.b[w13, 14:15], {z12.b, z13.b}  // 11000000-00000100-10100001-10000111
// CHECK-INST: mov     za0v.b[w13, 14:15], { z12.b, z13.b }
// CHECK-ENCODING: [0x87,0xa1,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004a187 <unknown>

// Aliases

mov     za0v.b[w12, 0:1], {z0.b, z1.b}  // 11000000-00000100-10000000-00000000
// CHECK-INST: mov     za0v.b[w12, 0:1], { z0.b, z1.b }
// CHECK-ENCODING: [0x00,0x80,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048000 <unknown>

mov     za0v.b[w14, 10:11], {z10.b, z11.b}  // 11000000-00000100-11000001-01000101
// CHECK-INST: mov     za0v.b[w14, 10:11], { z10.b, z11.b }
// CHECK-ENCODING: [0x45,0xc1,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004c145 <unknown>

mov     za0v.b[w15, 14:15], {z12.b, z13.b}  // 11000000-00000100-11100001-10000111
// CHECK-INST: mov     za0v.b[w15, 14:15], { z12.b, z13.b }
// CHECK-ENCODING: [0x87,0xe1,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004e187 <unknown>

mov     za0v.b[w15, 14:15], {z30.b, z31.b}  // 11000000-00000100-11100011-11000111
// CHECK-INST: mov     za0v.b[w15, 14:15], { z30.b, z31.b }
// CHECK-ENCODING: [0xc7,0xe3,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004e3c7 <unknown>

mov     za0v.b[w12, 10:11], {z16.b, z17.b}  // 11000000-00000100-10000010-00000101
// CHECK-INST: mov     za0v.b[w12, 10:11], { z16.b, z17.b }
// CHECK-ENCODING: [0x05,0x82,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048205 <unknown>

mov     za0v.b[w12, 2:3], {z0.b, z1.b}  // 11000000-00000100-10000000-00000001
// CHECK-INST: mov     za0v.b[w12, 2:3], { z0.b, z1.b }
// CHECK-ENCODING: [0x01,0x80,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048001 <unknown>

mov     za0v.b[w14, 0:1], {z18.b, z19.b}  // 11000000-00000100-11000010-01000000
// CHECK-INST: mov     za0v.b[w14, 0:1], { z18.b, z19.b }
// CHECK-ENCODING: [0x40,0xc2,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004c240 <unknown>

mov     za0v.b[w12, 0:1], {z12.b, z13.b}  // 11000000-00000100-10000001-10000000
// CHECK-INST: mov     za0v.b[w12, 0:1], { z12.b, z13.b }
// CHECK-ENCODING: [0x80,0x81,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048180 <unknown>

mov     za0v.b[w14, 2:3], {z0.b, z1.b}  // 11000000-00000100-11000000-00000001
// CHECK-INST: mov     za0v.b[w14, 2:3], { z0.b, z1.b }
// CHECK-ENCODING: [0x01,0xc0,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004c001 <unknown>

mov     za0v.b[w12, 10:11], {z22.b, z23.b}  // 11000000-00000100-10000010-11000101
// CHECK-INST: mov     za0v.b[w12, 10:11], { z22.b, z23.b }
// CHECK-ENCODING: [0xc5,0x82,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c00482c5 <unknown>

mov     za0v.b[w15, 4:5], {z8.b, z9.b}  // 11000000-00000100-11100001-00000010
// CHECK-INST: mov     za0v.b[w15, 4:5], { z8.b, z9.b }
// CHECK-ENCODING: [0x02,0xe1,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004e102 <unknown>

mov     za0v.b[w13, 14:15], {z12.b, z13.b}  // 11000000-00000100-10100001-10000111
// CHECK-INST: mov     za0v.b[w13, 14:15], { z12.b, z13.b }
// CHECK-ENCODING: [0x87,0xa1,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004a187 <unknown>


mova    {z0.h - z3.h}, za0h.h[w12, 0:3]  // 11000000-01000110-00000100-00000000
// CHECK-INST: mov     { z0.h - z3.h }, za0h.h[w12, 0:3]
// CHECK-ENCODING: [0x00,0x04,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0460400 <unknown>

mova    {z20.h - z23.h}, za1h.h[w14, 0:3]  // 11000000-01000110-01000100-01010100
// CHECK-INST: mov     { z20.h - z23.h }, za1h.h[w14, 0:3]
// CHECK-ENCODING: [0x54,0x44,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0464454 <unknown>

mova    {z20.h - z23.h}, za0h.h[w15, 4:7]  // 11000000-01000110-01100100-00110100
// CHECK-INST: mov     { z20.h - z23.h }, za0h.h[w15, 4:7]
// CHECK-ENCODING: [0x34,0x64,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0466434 <unknown>

mova    {z28.h - z31.h}, za1h.h[w15, 4:7]  // 11000000-01000110-01100100-01111100
// CHECK-INST: mov     { z28.h - z31.h }, za1h.h[w15, 4:7]
// CHECK-ENCODING: [0x7c,0x64,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046647c <unknown>

mova    {z4.h - z7.h}, za0h.h[w12, 4:7]  // 11000000-01000110-00000100-00100100
// CHECK-INST: mov     { z4.h - z7.h }, za0h.h[w12, 4:7]
// CHECK-ENCODING: [0x24,0x04,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0460424 <unknown>

mova    {z0.h - z3.h}, za0h.h[w12, 4:7]  // 11000000-01000110-00000100-00100000
// CHECK-INST: mov     { z0.h - z3.h }, za0h.h[w12, 4:7]
// CHECK-ENCODING: [0x20,0x04,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0460420 <unknown>

mova    {z24.h - z27.h}, za1h.h[w14, 4:7]  // 11000000-01000110-01000100-01111000
// CHECK-INST: mov     { z24.h - z27.h }, za1h.h[w14, 4:7]
// CHECK-ENCODING: [0x78,0x44,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0464478 <unknown>

mova    {z16.h - z19.h}, za0h.h[w14, 4:7]  // 11000000-01000110-01000100-00110000
// CHECK-INST: mov     { z16.h - z19.h }, za0h.h[w14, 4:7]
// CHECK-ENCODING: [0x30,0x44,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0464430 <unknown>

mova    {z28.h - z31.h}, za1h.h[w12, 0:3]  // 11000000-01000110-00000100-01011100
// CHECK-INST: mov     { z28.h - z31.h }, za1h.h[w12, 0:3]
// CHECK-ENCODING: [0x5c,0x04,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046045c <unknown>

mova    {z0.h - z3.h}, za0h.h[w15, 4:7]  // 11000000-01000110-01100100-00100000
// CHECK-INST: mov     { z0.h - z3.h }, za0h.h[w15, 4:7]
// CHECK-ENCODING: [0x20,0x64,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0466420 <unknown>

mova    {z4.h - z7.h}, za0h.h[w13, 0:3]  // 11000000-01000110-00100100-00000100
// CHECK-INST: mov     { z4.h - z7.h }, za0h.h[w13, 0:3]
// CHECK-ENCODING: [0x04,0x24,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0462404 <unknown>

// Aliases

mov     {z0.h - z3.h}, za0h.h[w12, 0:3]  // 11000000-01000110-00000100-00000000
// CHECK-INST: mov     { z0.h - z3.h }, za0h.h[w12, 0:3]
// CHECK-ENCODING: [0x00,0x04,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0460400 <unknown>

mov     {z20.h - z23.h}, za1h.h[w14, 0:3]  // 11000000-01000110-01000100-01010100
// CHECK-INST: mov     { z20.h - z23.h }, za1h.h[w14, 0:3]
// CHECK-ENCODING: [0x54,0x44,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0464454 <unknown>

mov     {z20.h - z23.h}, za0h.h[w15, 4:7]  // 11000000-01000110-01100100-00110100
// CHECK-INST: mov     { z20.h - z23.h }, za0h.h[w15, 4:7]
// CHECK-ENCODING: [0x34,0x64,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0466434 <unknown>

mov     {z28.h - z31.h}, za1h.h[w15, 4:7]  // 11000000-01000110-01100100-01111100
// CHECK-INST: mov     { z28.h - z31.h }, za1h.h[w15, 4:7]
// CHECK-ENCODING: [0x7c,0x64,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046647c <unknown>

mov     {z4.h - z7.h}, za0h.h[w12, 4:7]  // 11000000-01000110-00000100-00100100
// CHECK-INST: mov     { z4.h - z7.h }, za0h.h[w12, 4:7]
// CHECK-ENCODING: [0x24,0x04,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0460424 <unknown>

mov     {z0.h - z3.h}, za0h.h[w12, 4:7]  // 11000000-01000110-00000100-00100000
// CHECK-INST: mov     { z0.h - z3.h }, za0h.h[w12, 4:7]
// CHECK-ENCODING: [0x20,0x04,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0460420 <unknown>

mov     {z24.h - z27.h}, za1h.h[w14, 4:7]  // 11000000-01000110-01000100-01111000
// CHECK-INST: mov     { z24.h - z27.h }, za1h.h[w14, 4:7]
// CHECK-ENCODING: [0x78,0x44,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0464478 <unknown>

mov     {z16.h - z19.h}, za0h.h[w14, 4:7]  // 11000000-01000110-01000100-00110000
// CHECK-INST: mov     { z16.h - z19.h }, za0h.h[w14, 4:7]
// CHECK-ENCODING: [0x30,0x44,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0464430 <unknown>

mov     {z28.h - z31.h}, za1h.h[w12, 0:3]  // 11000000-01000110-00000100-01011100
// CHECK-INST: mov     { z28.h - z31.h }, za1h.h[w12, 0:3]
// CHECK-ENCODING: [0x5c,0x04,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046045c <unknown>

mov     {z0.h - z3.h}, za0h.h[w15, 4:7]  // 11000000-01000110-01100100-00100000
// CHECK-INST: mov     { z0.h - z3.h }, za0h.h[w15, 4:7]
// CHECK-ENCODING: [0x20,0x64,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0466420 <unknown>

mov     {z4.h - z7.h}, za0h.h[w13, 0:3]  // 11000000-01000110-00100100-00000100
// CHECK-INST: mov     { z4.h - z7.h }, za0h.h[w13, 0:3]
// CHECK-ENCODING: [0x04,0x24,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0462404 <unknown>


mova    {z0.h - z3.h}, za0v.h[w12, 0:3]  // 11000000-01000110-10000100-00000000
// CHECK-INST: mov     { z0.h - z3.h }, za0v.h[w12, 0:3]
// CHECK-ENCODING: [0x00,0x84,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0468400 <unknown>

mova    {z20.h - z23.h}, za1v.h[w14, 0:3]  // 11000000-01000110-11000100-01010100
// CHECK-INST: mov     { z20.h - z23.h }, za1v.h[w14, 0:3]
// CHECK-ENCODING: [0x54,0xc4,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046c454 <unknown>

mova    {z20.h - z23.h}, za0v.h[w15, 4:7]  // 11000000-01000110-11100100-00110100
// CHECK-INST: mov     { z20.h - z23.h }, za0v.h[w15, 4:7]
// CHECK-ENCODING: [0x34,0xe4,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046e434 <unknown>

mova    {z28.h - z31.h}, za1v.h[w15, 4:7]  // 11000000-01000110-11100100-01111100
// CHECK-INST: mov     { z28.h - z31.h }, za1v.h[w15, 4:7]
// CHECK-ENCODING: [0x7c,0xe4,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046e47c <unknown>

mova    {z4.h - z7.h}, za0v.h[w12, 4:7]  // 11000000-01000110-10000100-00100100
// CHECK-INST: mov     { z4.h - z7.h }, za0v.h[w12, 4:7]
// CHECK-ENCODING: [0x24,0x84,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0468424 <unknown>

mova    {z0.h - z3.h}, za0v.h[w12, 4:7]  // 11000000-01000110-10000100-00100000
// CHECK-INST: mov     { z0.h - z3.h }, za0v.h[w12, 4:7]
// CHECK-ENCODING: [0x20,0x84,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0468420 <unknown>

mova    {z24.h - z27.h}, za1v.h[w14, 4:7]  // 11000000-01000110-11000100-01111000
// CHECK-INST: mov     { z24.h - z27.h }, za1v.h[w14, 4:7]
// CHECK-ENCODING: [0x78,0xc4,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046c478 <unknown>

mova    {z16.h - z19.h}, za0v.h[w14, 4:7]  // 11000000-01000110-11000100-00110000
// CHECK-INST: mov     { z16.h - z19.h }, za0v.h[w14, 4:7]
// CHECK-ENCODING: [0x30,0xc4,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046c430 <unknown>

mova    {z28.h - z31.h}, za1v.h[w12, 0:3]  // 11000000-01000110-10000100-01011100
// CHECK-INST: mov     { z28.h - z31.h }, za1v.h[w12, 0:3]
// CHECK-ENCODING: [0x5c,0x84,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046845c <unknown>

mova    {z0.h - z3.h}, za0v.h[w15, 4:7]  // 11000000-01000110-11100100-00100000
// CHECK-INST: mov     { z0.h - z3.h }, za0v.h[w15, 4:7]
// CHECK-ENCODING: [0x20,0xe4,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046e420 <unknown>

mova    {z4.h - z7.h}, za0v.h[w13, 0:3]  // 11000000-01000110-10100100-00000100
// CHECK-INST: mov     { z4.h - z7.h }, za0v.h[w13, 0:3]
// CHECK-ENCODING: [0x04,0xa4,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046a404 <unknown>

// Aliases

mov     {z0.h - z3.h}, za0v.h[w12, 0:3]  // 11000000-01000110-10000100-00000000
// CHECK-INST: mov     { z0.h - z3.h }, za0v.h[w12, 0:3]
// CHECK-ENCODING: [0x00,0x84,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0468400 <unknown>

mov     {z20.h - z23.h}, za1v.h[w14, 0:3]  // 11000000-01000110-11000100-01010100
// CHECK-INST: mov     { z20.h - z23.h }, za1v.h[w14, 0:3]
// CHECK-ENCODING: [0x54,0xc4,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046c454 <unknown>

mov     {z20.h - z23.h}, za0v.h[w15, 4:7]  // 11000000-01000110-11100100-00110100
// CHECK-INST: mov     { z20.h - z23.h }, za0v.h[w15, 4:7]
// CHECK-ENCODING: [0x34,0xe4,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046e434 <unknown>

mov     {z28.h - z31.h}, za1v.h[w15, 4:7]  // 11000000-01000110-11100100-01111100
// CHECK-INST: mov     { z28.h - z31.h }, za1v.h[w15, 4:7]
// CHECK-ENCODING: [0x7c,0xe4,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046e47c <unknown>

mov     {z4.h - z7.h}, za0v.h[w12, 4:7]  // 11000000-01000110-10000100-00100100
// CHECK-INST: mov     { z4.h - z7.h }, za0v.h[w12, 4:7]
// CHECK-ENCODING: [0x24,0x84,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0468424 <unknown>

mov     {z0.h - z3.h}, za0v.h[w12, 4:7]  // 11000000-01000110-10000100-00100000
// CHECK-INST: mov     { z0.h - z3.h }, za0v.h[w12, 4:7]
// CHECK-ENCODING: [0x20,0x84,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0468420 <unknown>

mov     {z24.h - z27.h}, za1v.h[w14, 4:7]  // 11000000-01000110-11000100-01111000
// CHECK-INST: mov     { z24.h - z27.h }, za1v.h[w14, 4:7]
// CHECK-ENCODING: [0x78,0xc4,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046c478 <unknown>

mov     {z16.h - z19.h}, za0v.h[w14, 4:7]  // 11000000-01000110-11000100-00110000
// CHECK-INST: mov     { z16.h - z19.h }, za0v.h[w14, 4:7]
// CHECK-ENCODING: [0x30,0xc4,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046c430 <unknown>

mov     {z28.h - z31.h}, za1v.h[w12, 0:3]  // 11000000-01000110-10000100-01011100
// CHECK-INST: mov     { z28.h - z31.h }, za1v.h[w12, 0:3]
// CHECK-ENCODING: [0x5c,0x84,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046845c <unknown>

mov     {z0.h - z3.h}, za0v.h[w15, 4:7]  // 11000000-01000110-11100100-00100000
// CHECK-INST: mov     { z0.h - z3.h }, za0v.h[w15, 4:7]
// CHECK-ENCODING: [0x20,0xe4,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046e420 <unknown>

mov     {z4.h - z7.h}, za0v.h[w13, 0:3]  // 11000000-01000110-10100100-00000100
// CHECK-INST: mov     { z4.h - z7.h }, za0v.h[w13, 0:3]
// CHECK-ENCODING: [0x04,0xa4,0x46,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c046a404 <unknown>


mova    za0h.h[w12, 0:3], {z0.h - z3.h}  // 11000000-01000100-00000100-00000000
// CHECK-INST: mov     za0h.h[w12, 0:3], { z0.h - z3.h }
// CHECK-ENCODING: [0x00,0x04,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440400 <unknown>

mova    za0h.h[w14, 4:7], {z8.h - z11.h}  // 11000000-01000100-01000101-00000001
// CHECK-INST: mov     za0h.h[w14, 4:7], { z8.h - z11.h }
// CHECK-ENCODING: [0x01,0x45,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0444501 <unknown>

mova    za1h.h[w15, 4:7], {z12.h - z15.h}  // 11000000-01000100-01100101-10000011
// CHECK-INST: mov     za1h.h[w15, 4:7], { z12.h - z15.h }
// CHECK-ENCODING: [0x83,0x65,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0446583 <unknown>

mova    za1h.h[w15, 4:7], {z28.h - z31.h}  // 11000000-01000100-01100111-10000011
// CHECK-INST: mov     za1h.h[w15, 4:7], { z28.h - z31.h }
// CHECK-ENCODING: [0x83,0x67,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0446783 <unknown>

mova    za0h.h[w12, 4:7], {z16.h - z19.h}  // 11000000-01000100-00000110-00000001
// CHECK-INST: mov     za0h.h[w12, 4:7], { z16.h - z19.h }
// CHECK-ENCODING: [0x01,0x06,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440601 <unknown>

mova    za0h.h[w12, 4:7], {z0.h - z3.h}  // 11000000-01000100-00000100-00000001
// CHECK-INST: mov     za0h.h[w12, 4:7], { z0.h - z3.h }
// CHECK-ENCODING: [0x01,0x04,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440401 <unknown>

mova    za0h.h[w14, 0:3], {z16.h - z19.h}  // 11000000-01000100-01000110-00000000
// CHECK-INST: mov     za0h.h[w14, 0:3], { z16.h - z19.h }
// CHECK-ENCODING: [0x00,0x46,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0444600 <unknown>

mova    za0h.h[w12, 0:3], {z12.h - z15.h}  // 11000000-01000100-00000101-10000000
// CHECK-INST: mov     za0h.h[w12, 0:3], { z12.h - z15.h }
// CHECK-ENCODING: [0x80,0x05,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440580 <unknown>

mova    za0h.h[w14, 4:7], {z0.h - z3.h}  // 11000000-01000100-01000100-00000001
// CHECK-INST: mov     za0h.h[w14, 4:7], { z0.h - z3.h }
// CHECK-ENCODING: [0x01,0x44,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0444401 <unknown>

mova    za0h.h[w12, 4:7], {z20.h - z23.h}  // 11000000-01000100-00000110-10000001
// CHECK-INST: mov     za0h.h[w12, 4:7], { z20.h - z23.h }
// CHECK-ENCODING: [0x81,0x06,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440681 <unknown>

mova    za1h.h[w15, 0:3], {z8.h - z11.h}  // 11000000-01000100-01100101-00000010
// CHECK-INST: mov     za1h.h[w15, 0:3], { z8.h - z11.h }
// CHECK-ENCODING: [0x02,0x65,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0446502 <unknown>

mova    za1h.h[w13, 4:7], {z12.h - z15.h}  // 11000000-01000100-00100101-10000011
// CHECK-INST: mov     za1h.h[w13, 4:7], { z12.h - z15.h }
// CHECK-ENCODING: [0x83,0x25,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0442583 <unknown>

// Aliases

mov     za0h.h[w12, 0:3], {z0.h - z3.h}  // 11000000-01000100-00000100-00000000
// CHECK-INST: mov     za0h.h[w12, 0:3], { z0.h - z3.h }
// CHECK-ENCODING: [0x00,0x04,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440400 <unknown>

mov     za0h.h[w14, 4:7], {z8.h - z11.h}  // 11000000-01000100-01000101-00000001
// CHECK-INST: mov     za0h.h[w14, 4:7], { z8.h - z11.h }
// CHECK-ENCODING: [0x01,0x45,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0444501 <unknown>

mov     za1h.h[w15, 4:7], {z12.h - z15.h}  // 11000000-01000100-01100101-10000011
// CHECK-INST: mov     za1h.h[w15, 4:7], { z12.h - z15.h }
// CHECK-ENCODING: [0x83,0x65,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0446583 <unknown>

mov     za1h.h[w15, 4:7], {z28.h - z31.h}  // 11000000-01000100-01100111-10000011
// CHECK-INST: mov     za1h.h[w15, 4:7], { z28.h - z31.h }
// CHECK-ENCODING: [0x83,0x67,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0446783 <unknown>

mov     za0h.h[w12, 4:7], {z16.h - z19.h}  // 11000000-01000100-00000110-00000001
// CHECK-INST: mov     za0h.h[w12, 4:7], { z16.h - z19.h }
// CHECK-ENCODING: [0x01,0x06,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440601 <unknown>

mov     za0h.h[w12, 4:7], {z0.h - z3.h}  // 11000000-01000100-00000100-00000001
// CHECK-INST: mov     za0h.h[w12, 4:7], { z0.h - z3.h }
// CHECK-ENCODING: [0x01,0x04,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440401 <unknown>

mov     za0h.h[w14, 0:3], {z16.h - z19.h}  // 11000000-01000100-01000110-00000000
// CHECK-INST: mov     za0h.h[w14, 0:3], { z16.h - z19.h }
// CHECK-ENCODING: [0x00,0x46,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0444600 <unknown>

mov     za0h.h[w12, 0:3], {z12.h - z15.h}  // 11000000-01000100-00000101-10000000
// CHECK-INST: mov     za0h.h[w12, 0:3], { z12.h - z15.h }
// CHECK-ENCODING: [0x80,0x05,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440580 <unknown>

mov     za0h.h[w14, 4:7], {z0.h - z3.h}  // 11000000-01000100-01000100-00000001
// CHECK-INST: mov     za0h.h[w14, 4:7], { z0.h - z3.h }
// CHECK-ENCODING: [0x01,0x44,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0444401 <unknown>

mov     za0h.h[w12, 4:7], {z20.h - z23.h}  // 11000000-01000100-00000110-10000001
// CHECK-INST: mov     za0h.h[w12, 4:7], { z20.h - z23.h }
// CHECK-ENCODING: [0x81,0x06,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0440681 <unknown>

mov     za1h.h[w15, 0:3], {z8.h - z11.h}  // 11000000-01000100-01100101-00000010
// CHECK-INST: mov     za1h.h[w15, 0:3], { z8.h - z11.h }
// CHECK-ENCODING: [0x02,0x65,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0446502 <unknown>

mov     za1h.h[w13, 4:7], {z12.h - z15.h}  // 11000000-01000100-00100101-10000011
// CHECK-INST: mov     za1h.h[w13, 4:7], { z12.h - z15.h }
// CHECK-ENCODING: [0x83,0x25,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0442583 <unknown>


mova    za0v.h[w12, 0:3], {z0.h - z3.h}  // 11000000-01000100-10000100-00000000
// CHECK-INST: mov     za0v.h[w12, 0:3], { z0.h - z3.h }
// CHECK-ENCODING: [0x00,0x84,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448400 <unknown>

mova    za0v.h[w14, 4:7], {z8.h - z11.h}  // 11000000-01000100-11000101-00000001
// CHECK-INST: mov     za0v.h[w14, 4:7], { z8.h - z11.h }
// CHECK-ENCODING: [0x01,0xc5,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044c501 <unknown>

mova    za1v.h[w15, 4:7], {z12.h - z15.h}  // 11000000-01000100-11100101-10000011
// CHECK-INST: mov     za1v.h[w15, 4:7], { z12.h - z15.h }
// CHECK-ENCODING: [0x83,0xe5,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044e583 <unknown>

mova    za1v.h[w15, 4:7], {z28.h - z31.h}  // 11000000-01000100-11100111-10000011
// CHECK-INST: mov     za1v.h[w15, 4:7], { z28.h - z31.h }
// CHECK-ENCODING: [0x83,0xe7,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044e783 <unknown>

mova    za0v.h[w12, 4:7], {z16.h - z19.h}  // 11000000-01000100-10000110-00000001
// CHECK-INST: mov     za0v.h[w12, 4:7], { z16.h - z19.h }
// CHECK-ENCODING: [0x01,0x86,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448601 <unknown>

mova    za0v.h[w12, 4:7], {z0.h - z3.h}  // 11000000-01000100-10000100-00000001
// CHECK-INST: mov     za0v.h[w12, 4:7], { z0.h - z3.h }
// CHECK-ENCODING: [0x01,0x84,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448401 <unknown>

mova    za0v.h[w14, 0:3], {z16.h - z19.h}  // 11000000-01000100-11000110-00000000
// CHECK-INST: mov     za0v.h[w14, 0:3], { z16.h - z19.h }
// CHECK-ENCODING: [0x00,0xc6,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044c600 <unknown>

mova    za0v.h[w12, 0:3], {z12.h - z15.h}  // 11000000-01000100-10000101-10000000
// CHECK-INST: mov     za0v.h[w12, 0:3], { z12.h - z15.h }
// CHECK-ENCODING: [0x80,0x85,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448580 <unknown>

mova    za0v.h[w14, 4:7], {z0.h - z3.h}  // 11000000-01000100-11000100-00000001
// CHECK-INST: mov     za0v.h[w14, 4:7], { z0.h - z3.h }
// CHECK-ENCODING: [0x01,0xc4,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044c401 <unknown>

mova    za0v.h[w12, 4:7], {z20.h - z23.h}  // 11000000-01000100-10000110-10000001
// CHECK-INST: mov     za0v.h[w12, 4:7], { z20.h - z23.h }
// CHECK-ENCODING: [0x81,0x86,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448681 <unknown>

mova    za1v.h[w15, 0:3], {z8.h - z11.h}  // 11000000-01000100-11100101-00000010
// CHECK-INST: mov     za1v.h[w15, 0:3], { z8.h - z11.h }
// CHECK-ENCODING: [0x02,0xe5,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044e502 <unknown>

mova    za1v.h[w13, 4:7], {z12.h - z15.h}  // 11000000-01000100-10100101-10000011
// CHECK-INST: mov     za1v.h[w13, 4:7], { z12.h - z15.h }
// CHECK-ENCODING: [0x83,0xa5,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044a583 <unknown>

// Aliases

mov     za0v.h[w12, 0:3], {z0.h - z3.h}  // 11000000-01000100-10000100-00000000
// CHECK-INST: mov     za0v.h[w12, 0:3], { z0.h - z3.h }
// CHECK-ENCODING: [0x00,0x84,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448400 <unknown>

mov     za0v.h[w14, 4:7], {z8.h - z11.h}  // 11000000-01000100-11000101-00000001
// CHECK-INST: mov     za0v.h[w14, 4:7], { z8.h - z11.h }
// CHECK-ENCODING: [0x01,0xc5,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044c501 <unknown>

mov     za1v.h[w15, 4:7], {z12.h - z15.h}  // 11000000-01000100-11100101-10000011
// CHECK-INST: mov     za1v.h[w15, 4:7], { z12.h - z15.h }
// CHECK-ENCODING: [0x83,0xe5,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044e583 <unknown>

mov     za1v.h[w15, 4:7], {z28.h - z31.h}  // 11000000-01000100-11100111-10000011
// CHECK-INST: mov     za1v.h[w15, 4:7], { z28.h - z31.h }
// CHECK-ENCODING: [0x83,0xe7,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044e783 <unknown>

mov     za0v.h[w12, 4:7], {z16.h - z19.h}  // 11000000-01000100-10000110-00000001
// CHECK-INST: mov     za0v.h[w12, 4:7], { z16.h - z19.h }
// CHECK-ENCODING: [0x01,0x86,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448601 <unknown>

mov     za0v.h[w12, 4:7], {z0.h - z3.h}  // 11000000-01000100-10000100-00000001
// CHECK-INST: mov     za0v.h[w12, 4:7], { z0.h - z3.h }
// CHECK-ENCODING: [0x01,0x84,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448401 <unknown>

mov     za0v.h[w14, 0:3], {z16.h - z19.h}  // 11000000-01000100-11000110-00000000
// CHECK-INST: mov     za0v.h[w14, 0:3], { z16.h - z19.h }
// CHECK-ENCODING: [0x00,0xc6,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044c600 <unknown>

mov     za0v.h[w12, 0:3], {z12.h - z15.h}  // 11000000-01000100-10000101-10000000
// CHECK-INST: mov     za0v.h[w12, 0:3], { z12.h - z15.h }
// CHECK-ENCODING: [0x80,0x85,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448580 <unknown>

mov     za0v.h[w14, 4:7], {z0.h - z3.h}  // 11000000-01000100-11000100-00000001
// CHECK-INST: mov     za0v.h[w14, 4:7], { z0.h - z3.h }
// CHECK-ENCODING: [0x01,0xc4,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044c401 <unknown>

mov     za0v.h[w12, 4:7], {z20.h - z23.h}  // 11000000-01000100-10000110-10000001
// CHECK-INST: mov     za0v.h[w12, 4:7], { z20.h - z23.h }
// CHECK-ENCODING: [0x81,0x86,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0448681 <unknown>

mov     za1v.h[w15, 0:3], {z8.h - z11.h}  // 11000000-01000100-11100101-00000010
// CHECK-INST: mov     za1v.h[w15, 0:3], { z8.h - z11.h }
// CHECK-ENCODING: [0x02,0xe5,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044e502 <unknown>

mov     za1v.h[w13, 4:7], {z12.h - z15.h}  // 11000000-01000100-10100101-10000011
// CHECK-INST: mov     za1v.h[w13, 4:7], { z12.h - z15.h }
// CHECK-ENCODING: [0x83,0xa5,0x44,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c044a583 <unknown>


mova    {z0.s - z3.s}, za0h.s[w12, 0:3]  // 11000000-10000110-00000100-00000000
// CHECK-INST: mov     { z0.s - z3.s }, za0h.s[w12, 0:3]
// CHECK-ENCODING: [0x00,0x04,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0860400 <unknown>

mova    {z20.s - z23.s}, za2h.s[w14, 0:3]  // 11000000-10000110-01000100-01010100
// CHECK-INST: mov     { z20.s - z23.s }, za2h.s[w14, 0:3]
// CHECK-ENCODING: [0x54,0x44,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0864454 <unknown>

mova    {z20.s - z23.s}, za1h.s[w15, 0:3]  // 11000000-10000110-01100100-00110100
// CHECK-INST: mov     { z20.s - z23.s }, za1h.s[w15, 0:3]
// CHECK-ENCODING: [0x34,0x64,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0866434 <unknown>

mova    {z28.s - z31.s}, za3h.s[w15, 0:3]  // 11000000-10000110-01100100-01111100
// CHECK-INST: mov     { z28.s - z31.s }, za3h.s[w15, 0:3]
// CHECK-ENCODING: [0x7c,0x64,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086647c <unknown>

mova    {z4.s - z7.s}, za1h.s[w12, 0:3]  // 11000000-10000110-00000100-00100100
// CHECK-INST: mov     { z4.s - z7.s }, za1h.s[w12, 0:3]
// CHECK-ENCODING: [0x24,0x04,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0860424 <unknown>

mova    {z0.s - z3.s}, za1h.s[w12, 0:3]  // 11000000-10000110-00000100-00100000
// CHECK-INST: mov     { z0.s - z3.s }, za1h.s[w12, 0:3]
// CHECK-ENCODING: [0x20,0x04,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0860420 <unknown>

mova    {z24.s - z27.s}, za3h.s[w14, 0:3]  // 11000000-10000110-01000100-01111000
// CHECK-INST: mov     { z24.s - z27.s }, za3h.s[w14, 0:3]
// CHECK-ENCODING: [0x78,0x44,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0864478 <unknown>

mova    {z16.s - z19.s}, za1h.s[w14, 0:3]  // 11000000-10000110-01000100-00110000
// CHECK-INST: mov     { z16.s - z19.s }, za1h.s[w14, 0:3]
// CHECK-ENCODING: [0x30,0x44,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0864430 <unknown>

mova    {z28.s - z31.s}, za2h.s[w12, 0:3]  // 11000000-10000110-00000100-01011100
// CHECK-INST: mov     { z28.s - z31.s }, za2h.s[w12, 0:3]
// CHECK-ENCODING: [0x5c,0x04,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086045c <unknown>

mova    {z0.s - z3.s}, za1h.s[w15, 0:3]  // 11000000-10000110-01100100-00100000
// CHECK-INST: mov     { z0.s - z3.s }, za1h.s[w15, 0:3]
// CHECK-ENCODING: [0x20,0x64,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0866420 <unknown>

mova    {z4.s - z7.s}, za0h.s[w13, 0:3]  // 11000000-10000110-00100100-00000100
// CHECK-INST: mov     { z4.s - z7.s }, za0h.s[w13, 0:3]
// CHECK-ENCODING: [0x04,0x24,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0862404 <unknown>

// Aliases

mov     {z0.s - z3.s}, za0h.s[w12, 0:3]  // 11000000-10000110-00000100-00000000
// CHECK-INST: mov     { z0.s - z3.s }, za0h.s[w12, 0:3]
// CHECK-ENCODING: [0x00,0x04,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0860400 <unknown>

mov     {z20.s - z23.s}, za2h.s[w14, 0:3]  // 11000000-10000110-01000100-01010100
// CHECK-INST: mov     { z20.s - z23.s }, za2h.s[w14, 0:3]
// CHECK-ENCODING: [0x54,0x44,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0864454 <unknown>

mov     {z20.s - z23.s}, za1h.s[w15, 0:3]  // 11000000-10000110-01100100-00110100
// CHECK-INST: mov     { z20.s - z23.s }, za1h.s[w15, 0:3]
// CHECK-ENCODING: [0x34,0x64,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0866434 <unknown>

mov     {z28.s - z31.s}, za3h.s[w15, 0:3]  // 11000000-10000110-01100100-01111100
// CHECK-INST: mov     { z28.s - z31.s }, za3h.s[w15, 0:3]
// CHECK-ENCODING: [0x7c,0x64,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086647c <unknown>

mov     {z4.s - z7.s}, za1h.s[w12, 0:3]  // 11000000-10000110-00000100-00100100
// CHECK-INST: mov     { z4.s - z7.s }, za1h.s[w12, 0:3]
// CHECK-ENCODING: [0x24,0x04,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0860424 <unknown>

mov     {z0.s - z3.s}, za1h.s[w12, 0:3]  // 11000000-10000110-00000100-00100000
// CHECK-INST: mov     { z0.s - z3.s }, za1h.s[w12, 0:3]
// CHECK-ENCODING: [0x20,0x04,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0860420 <unknown>

mov     {z24.s - z27.s}, za3h.s[w14, 0:3]  // 11000000-10000110-01000100-01111000
// CHECK-INST: mov     { z24.s - z27.s }, za3h.s[w14, 0:3]
// CHECK-ENCODING: [0x78,0x44,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0864478 <unknown>

mov     {z16.s - z19.s}, za1h.s[w14, 0:3]  // 11000000-10000110-01000100-00110000
// CHECK-INST: mov     { z16.s - z19.s }, za1h.s[w14, 0:3]
// CHECK-ENCODING: [0x30,0x44,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0864430 <unknown>

mov     {z28.s - z31.s}, za2h.s[w12, 0:3]  // 11000000-10000110-00000100-01011100
// CHECK-INST: mov     { z28.s - z31.s }, za2h.s[w12, 0:3]
// CHECK-ENCODING: [0x5c,0x04,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086045c <unknown>

mov     {z0.s - z3.s}, za1h.s[w15, 0:3]  // 11000000-10000110-01100100-00100000
// CHECK-INST: mov     { z0.s - z3.s }, za1h.s[w15, 0:3]
// CHECK-ENCODING: [0x20,0x64,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0866420 <unknown>

mov     {z4.s - z7.s}, za0h.s[w13, 0:3]  // 11000000-10000110-00100100-00000100
// CHECK-INST: mov     { z4.s - z7.s }, za0h.s[w13, 0:3]
// CHECK-ENCODING: [0x04,0x24,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0862404 <unknown>


mova    {z0.s - z3.s}, za0v.s[w12, 0:3]  // 11000000-10000110-10000100-00000000
// CHECK-INST: mov     { z0.s - z3.s }, za0v.s[w12, 0:3]
// CHECK-ENCODING: [0x00,0x84,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0868400 <unknown>

mova    {z20.s - z23.s}, za2v.s[w14, 0:3]  // 11000000-10000110-11000100-01010100
// CHECK-INST: mov     { z20.s - z23.s }, za2v.s[w14, 0:3]
// CHECK-ENCODING: [0x54,0xc4,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086c454 <unknown>

mova    {z20.s - z23.s}, za1v.s[w15, 0:3]  // 11000000-10000110-11100100-00110100
// CHECK-INST: mov     { z20.s - z23.s }, za1v.s[w15, 0:3]
// CHECK-ENCODING: [0x34,0xe4,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086e434 <unknown>

mova    {z28.s - z31.s}, za3v.s[w15, 0:3]  // 11000000-10000110-11100100-01111100
// CHECK-INST: mov     { z28.s - z31.s }, za3v.s[w15, 0:3]
// CHECK-ENCODING: [0x7c,0xe4,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086e47c <unknown>

mova    {z4.s - z7.s}, za1v.s[w12, 0:3]  // 11000000-10000110-10000100-00100100
// CHECK-INST: mov     { z4.s - z7.s }, za1v.s[w12, 0:3]
// CHECK-ENCODING: [0x24,0x84,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0868424 <unknown>

mova    {z0.s - z3.s}, za1v.s[w12, 0:3]  // 11000000-10000110-10000100-00100000
// CHECK-INST: mov     { z0.s - z3.s }, za1v.s[w12, 0:3]
// CHECK-ENCODING: [0x20,0x84,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0868420 <unknown>

mova    {z24.s - z27.s}, za3v.s[w14, 0:3]  // 11000000-10000110-11000100-01111000
// CHECK-INST: mov     { z24.s - z27.s }, za3v.s[w14, 0:3]
// CHECK-ENCODING: [0x78,0xc4,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086c478 <unknown>

mova    {z16.s - z19.s}, za1v.s[w14, 0:3]  // 11000000-10000110-11000100-00110000
// CHECK-INST: mov     { z16.s - z19.s }, za1v.s[w14, 0:3]
// CHECK-ENCODING: [0x30,0xc4,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086c430 <unknown>

mova    {z28.s - z31.s}, za2v.s[w12, 0:3]  // 11000000-10000110-10000100-01011100
// CHECK-INST: mov     { z28.s - z31.s }, za2v.s[w12, 0:3]
// CHECK-ENCODING: [0x5c,0x84,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086845c <unknown>

mova    {z0.s - z3.s}, za1v.s[w15, 0:3]  // 11000000-10000110-11100100-00100000
// CHECK-INST: mov     { z0.s - z3.s }, za1v.s[w15, 0:3]
// CHECK-ENCODING: [0x20,0xe4,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086e420 <unknown>

mova    {z4.s - z7.s}, za0v.s[w13, 0:3]  // 11000000-10000110-10100100-00000100
// CHECK-INST: mov     { z4.s - z7.s }, za0v.s[w13, 0:3]
// CHECK-ENCODING: [0x04,0xa4,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086a404 <unknown>

// Aliases

mov     {z0.s - z3.s}, za0v.s[w12, 0:3]  // 11000000-10000110-10000100-00000000
// CHECK-INST: mov     { z0.s - z3.s }, za0v.s[w12, 0:3]
// CHECK-ENCODING: [0x00,0x84,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0868400 <unknown>

mov     {z20.s - z23.s}, za2v.s[w14, 0:3]  // 11000000-10000110-11000100-01010100
// CHECK-INST: mov     { z20.s - z23.s }, za2v.s[w14, 0:3]
// CHECK-ENCODING: [0x54,0xc4,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086c454 <unknown>

mov     {z20.s - z23.s}, za1v.s[w15, 0:3]  // 11000000-10000110-11100100-00110100
// CHECK-INST: mov     { z20.s - z23.s }, za1v.s[w15, 0:3]
// CHECK-ENCODING: [0x34,0xe4,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086e434 <unknown>

mov     {z28.s - z31.s}, za3v.s[w15, 0:3]  // 11000000-10000110-11100100-01111100
// CHECK-INST: mov     { z28.s - z31.s }, za3v.s[w15, 0:3]
// CHECK-ENCODING: [0x7c,0xe4,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086e47c <unknown>

mov     {z4.s - z7.s}, za1v.s[w12, 0:3]  // 11000000-10000110-10000100-00100100
// CHECK-INST: mov     { z4.s - z7.s }, za1v.s[w12, 0:3]
// CHECK-ENCODING: [0x24,0x84,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0868424 <unknown>

mov     {z0.s - z3.s}, za1v.s[w12, 0:3]  // 11000000-10000110-10000100-00100000
// CHECK-INST: mov     { z0.s - z3.s }, za1v.s[w12, 0:3]
// CHECK-ENCODING: [0x20,0x84,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0868420 <unknown>

mov     {z24.s - z27.s}, za3v.s[w14, 0:3]  // 11000000-10000110-11000100-01111000
// CHECK-INST: mov     { z24.s - z27.s }, za3v.s[w14, 0:3]
// CHECK-ENCODING: [0x78,0xc4,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086c478 <unknown>

mov     {z16.s - z19.s}, za1v.s[w14, 0:3]  // 11000000-10000110-11000100-00110000
// CHECK-INST: mov     { z16.s - z19.s }, za1v.s[w14, 0:3]
// CHECK-ENCODING: [0x30,0xc4,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086c430 <unknown>

mov     {z28.s - z31.s}, za2v.s[w12, 0:3]  // 11000000-10000110-10000100-01011100
// CHECK-INST: mov     { z28.s - z31.s }, za2v.s[w12, 0:3]
// CHECK-ENCODING: [0x5c,0x84,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086845c <unknown>

mov     {z0.s - z3.s}, za1v.s[w15, 0:3]  // 11000000-10000110-11100100-00100000
// CHECK-INST: mov     { z0.s - z3.s }, za1v.s[w15, 0:3]
// CHECK-ENCODING: [0x20,0xe4,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086e420 <unknown>

mov     {z4.s - z7.s}, za0v.s[w13, 0:3]  // 11000000-10000110-10100100-00000100
// CHECK-INST: mov     { z4.s - z7.s }, za0v.s[w13, 0:3]
// CHECK-ENCODING: [0x04,0xa4,0x86,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c086a404 <unknown>


mova    za0h.s[w12, 0:3], {z0.s - z3.s}  // 11000000-10000100-00000100-00000000
// CHECK-INST: mov     za0h.s[w12, 0:3], { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0x04,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840400 <unknown>

mova    za1h.s[w14, 0:3], {z8.s - z11.s}  // 11000000-10000100-01000101-00000001
// CHECK-INST: mov     za1h.s[w14, 0:3], { z8.s - z11.s }
// CHECK-ENCODING: [0x01,0x45,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0844501 <unknown>

mova    za3h.s[w15, 0:3], {z12.s - z15.s}  // 11000000-10000100-01100101-10000011
// CHECK-INST: mov     za3h.s[w15, 0:3], { z12.s - z15.s }
// CHECK-ENCODING: [0x83,0x65,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0846583 <unknown>

mova    za3h.s[w15, 0:3], {z28.s - z31.s}  // 11000000-10000100-01100111-10000011
// CHECK-INST: mov     za3h.s[w15, 0:3], { z28.s - z31.s }
// CHECK-ENCODING: [0x83,0x67,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0846783 <unknown>

mova    za1h.s[w12, 0:3], {z16.s - z19.s}  // 11000000-10000100-00000110-00000001
// CHECK-INST: mov     za1h.s[w12, 0:3], { z16.s - z19.s }
// CHECK-ENCODING: [0x01,0x06,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840601 <unknown>

mova    za1h.s[w12, 0:3], {z0.s - z3.s}  // 11000000-10000100-00000100-00000001
// CHECK-INST: mov     za1h.s[w12, 0:3], { z0.s - z3.s }
// CHECK-ENCODING: [0x01,0x04,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840401 <unknown>

mova    za0h.s[w14, 0:3], {z16.s - z19.s}  // 11000000-10000100-01000110-00000000
// CHECK-INST: mov     za0h.s[w14, 0:3], { z16.s - z19.s }
// CHECK-ENCODING: [0x00,0x46,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0844600 <unknown>

mova    za0h.s[w12, 0:3], {z12.s - z15.s}  // 11000000-10000100-00000101-10000000
// CHECK-INST: mov     za0h.s[w12, 0:3], { z12.s - z15.s }
// CHECK-ENCODING: [0x80,0x05,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840580 <unknown>

mova    za1h.s[w14, 0:3], {z0.s - z3.s}  // 11000000-10000100-01000100-00000001
// CHECK-INST: mov     za1h.s[w14, 0:3], { z0.s - z3.s }
// CHECK-ENCODING: [0x01,0x44,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0844401 <unknown>

mova    za1h.s[w12, 0:3], {z20.s - z23.s}  // 11000000-10000100-00000110-10000001
// CHECK-INST: mov     za1h.s[w12, 0:3], { z20.s - z23.s }
// CHECK-ENCODING: [0x81,0x06,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840681 <unknown>

mova    za2h.s[w15, 0:3], {z8.s - z11.s}  // 11000000-10000100-01100101-00000010
// CHECK-INST: mov     za2h.s[w15, 0:3], { z8.s - z11.s }
// CHECK-ENCODING: [0x02,0x65,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0846502 <unknown>

mova    za3h.s[w13, 0:3], {z12.s - z15.s}  // 11000000-10000100-00100101-10000011
// CHECK-INST: mov     za3h.s[w13, 0:3], { z12.s - z15.s }
// CHECK-ENCODING: [0x83,0x25,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0842583 <unknown>

// Aliases

mov     za0h.s[w12, 0:3], {z0.s - z3.s}  // 11000000-10000100-00000100-00000000
// CHECK-INST: mov     za0h.s[w12, 0:3], { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0x04,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840400 <unknown>

mov     za1h.s[w14, 0:3], {z8.s - z11.s}  // 11000000-10000100-01000101-00000001
// CHECK-INST: mov     za1h.s[w14, 0:3], { z8.s - z11.s }
// CHECK-ENCODING: [0x01,0x45,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0844501 <unknown>

mov     za3h.s[w15, 0:3], {z12.s - z15.s}  // 11000000-10000100-01100101-10000011
// CHECK-INST: mov     za3h.s[w15, 0:3], { z12.s - z15.s }
// CHECK-ENCODING: [0x83,0x65,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0846583 <unknown>

mov     za3h.s[w15, 0:3], {z28.s - z31.s}  // 11000000-10000100-01100111-10000011
// CHECK-INST: mov     za3h.s[w15, 0:3], { z28.s - z31.s }
// CHECK-ENCODING: [0x83,0x67,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0846783 <unknown>

mov     za1h.s[w12, 0:3], {z16.s - z19.s}  // 11000000-10000100-00000110-00000001
// CHECK-INST: mov     za1h.s[w12, 0:3], { z16.s - z19.s }
// CHECK-ENCODING: [0x01,0x06,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840601 <unknown>

mov     za1h.s[w12, 0:3], {z0.s - z3.s}  // 11000000-10000100-00000100-00000001
// CHECK-INST: mov     za1h.s[w12, 0:3], { z0.s - z3.s }
// CHECK-ENCODING: [0x01,0x04,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840401 <unknown>

mov     za0h.s[w14, 0:3], {z16.s - z19.s}  // 11000000-10000100-01000110-00000000
// CHECK-INST: mov     za0h.s[w14, 0:3], { z16.s - z19.s }
// CHECK-ENCODING: [0x00,0x46,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0844600 <unknown>

mov     za0h.s[w12, 0:3], {z12.s - z15.s}  // 11000000-10000100-00000101-10000000
// CHECK-INST: mov     za0h.s[w12, 0:3], { z12.s - z15.s }
// CHECK-ENCODING: [0x80,0x05,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840580 <unknown>

mov     za1h.s[w14, 0:3], {z0.s - z3.s}  // 11000000-10000100-01000100-00000001
// CHECK-INST: mov     za1h.s[w14, 0:3], { z0.s - z3.s }
// CHECK-ENCODING: [0x01,0x44,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0844401 <unknown>

mov     za1h.s[w12, 0:3], {z20.s - z23.s}  // 11000000-10000100-00000110-10000001
// CHECK-INST: mov     za1h.s[w12, 0:3], { z20.s - z23.s }
// CHECK-ENCODING: [0x81,0x06,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0840681 <unknown>

mov     za2h.s[w15, 0:3], {z8.s - z11.s}  // 11000000-10000100-01100101-00000010
// CHECK-INST: mov     za2h.s[w15, 0:3], { z8.s - z11.s }
// CHECK-ENCODING: [0x02,0x65,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0846502 <unknown>

mov     za3h.s[w13, 0:3], {z12.s - z15.s}  // 11000000-10000100-00100101-10000011
// CHECK-INST: mov     za3h.s[w13, 0:3], { z12.s - z15.s }
// CHECK-ENCODING: [0x83,0x25,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0842583 <unknown>


mova    za0v.s[w12, 0:3], {z0.s - z3.s}  // 11000000-10000100-10000100-00000000
// CHECK-INST: mov     za0v.s[w12, 0:3], { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0x84,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848400 <unknown>

mova    za1v.s[w14, 0:3], {z8.s - z11.s}  // 11000000-10000100-11000101-00000001
// CHECK-INST: mov     za1v.s[w14, 0:3], { z8.s - z11.s }
// CHECK-ENCODING: [0x01,0xc5,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084c501 <unknown>

mova    za3v.s[w15, 0:3], {z12.s - z15.s}  // 11000000-10000100-11100101-10000011
// CHECK-INST: mov     za3v.s[w15, 0:3], { z12.s - z15.s }
// CHECK-ENCODING: [0x83,0xe5,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084e583 <unknown>

mova    za3v.s[w15, 0:3], {z28.s - z31.s}  // 11000000-10000100-11100111-10000011
// CHECK-INST: mov     za3v.s[w15, 0:3], { z28.s - z31.s }
// CHECK-ENCODING: [0x83,0xe7,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084e783 <unknown>

mova    za1v.s[w12, 0:3], {z16.s - z19.s}  // 11000000-10000100-10000110-00000001
// CHECK-INST: mov     za1v.s[w12, 0:3], { z16.s - z19.s }
// CHECK-ENCODING: [0x01,0x86,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848601 <unknown>

mova    za1v.s[w12, 0:3], {z0.s - z3.s}  // 11000000-10000100-10000100-00000001
// CHECK-INST: mov     za1v.s[w12, 0:3], { z0.s - z3.s }
// CHECK-ENCODING: [0x01,0x84,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848401 <unknown>

mova    za0v.s[w14, 0:3], {z16.s - z19.s}  // 11000000-10000100-11000110-00000000
// CHECK-INST: mov     za0v.s[w14, 0:3], { z16.s - z19.s }
// CHECK-ENCODING: [0x00,0xc6,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084c600 <unknown>

mova    za0v.s[w12, 0:3], {z12.s - z15.s}  // 11000000-10000100-10000101-10000000
// CHECK-INST: mov     za0v.s[w12, 0:3], { z12.s - z15.s }
// CHECK-ENCODING: [0x80,0x85,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848580 <unknown>

mova    za1v.s[w14, 0:3], {z0.s - z3.s}  // 11000000-10000100-11000100-00000001
// CHECK-INST: mov     za1v.s[w14, 0:3], { z0.s - z3.s }
// CHECK-ENCODING: [0x01,0xc4,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084c401 <unknown>

mova    za1v.s[w12, 0:3], {z20.s - z23.s}  // 11000000-10000100-10000110-10000001
// CHECK-INST: mov     za1v.s[w12, 0:3], { z20.s - z23.s }
// CHECK-ENCODING: [0x81,0x86,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848681 <unknown>

mova    za2v.s[w15, 0:3], {z8.s - z11.s}  // 11000000-10000100-11100101-00000010
// CHECK-INST: mov     za2v.s[w15, 0:3], { z8.s - z11.s }
// CHECK-ENCODING: [0x02,0xe5,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084e502 <unknown>

mova    za3v.s[w13, 0:3], {z12.s - z15.s}  // 11000000-10000100-10100101-10000011
// CHECK-INST: mov     za3v.s[w13, 0:3], { z12.s - z15.s }
// CHECK-ENCODING: [0x83,0xa5,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084a583 <unknown>

// Aliases

mov     za0v.s[w12, 0:3], {z0.s - z3.s}  // 11000000-10000100-10000100-00000000
// CHECK-INST: mov     za0v.s[w12, 0:3], { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0x84,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848400 <unknown>

mov     za1v.s[w14, 0:3], {z8.s - z11.s}  // 11000000-10000100-11000101-00000001
// CHECK-INST: mov     za1v.s[w14, 0:3], { z8.s - z11.s }
// CHECK-ENCODING: [0x01,0xc5,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084c501 <unknown>

mov     za3v.s[w15, 0:3], {z12.s - z15.s}  // 11000000-10000100-11100101-10000011
// CHECK-INST: mov     za3v.s[w15, 0:3], { z12.s - z15.s }
// CHECK-ENCODING: [0x83,0xe5,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084e583 <unknown>

mov     za3v.s[w15, 0:3], {z28.s - z31.s}  // 11000000-10000100-11100111-10000011
// CHECK-INST: mov     za3v.s[w15, 0:3], { z28.s - z31.s }
// CHECK-ENCODING: [0x83,0xe7,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084e783 <unknown>

mov     za1v.s[w12, 0:3], {z16.s - z19.s}  // 11000000-10000100-10000110-00000001
// CHECK-INST: mov     za1v.s[w12, 0:3], { z16.s - z19.s }
// CHECK-ENCODING: [0x01,0x86,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848601 <unknown>

mov     za1v.s[w12, 0:3], {z0.s - z3.s}  // 11000000-10000100-10000100-00000001
// CHECK-INST: mov     za1v.s[w12, 0:3], { z0.s - z3.s }
// CHECK-ENCODING: [0x01,0x84,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848401 <unknown>

mov     za0v.s[w14, 0:3], {z16.s - z19.s}  // 11000000-10000100-11000110-00000000
// CHECK-INST: mov     za0v.s[w14, 0:3], { z16.s - z19.s }
// CHECK-ENCODING: [0x00,0xc6,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084c600 <unknown>

mov     za0v.s[w12, 0:3], {z12.s - z15.s}  // 11000000-10000100-10000101-10000000
// CHECK-INST: mov     za0v.s[w12, 0:3], { z12.s - z15.s }
// CHECK-ENCODING: [0x80,0x85,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848580 <unknown>

mov     za1v.s[w14, 0:3], {z0.s - z3.s}  // 11000000-10000100-11000100-00000001
// CHECK-INST: mov     za1v.s[w14, 0:3], { z0.s - z3.s }
// CHECK-ENCODING: [0x01,0xc4,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084c401 <unknown>

mov     za1v.s[w12, 0:3], {z20.s - z23.s}  // 11000000-10000100-10000110-10000001
// CHECK-INST: mov     za1v.s[w12, 0:3], { z20.s - z23.s }
// CHECK-ENCODING: [0x81,0x86,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0848681 <unknown>

mov     za2v.s[w15, 0:3], {z8.s - z11.s}  // 11000000-10000100-11100101-00000010
// CHECK-INST: mov     za2v.s[w15, 0:3], { z8.s - z11.s }
// CHECK-ENCODING: [0x02,0xe5,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084e502 <unknown>

mov     za3v.s[w13, 0:3], {z12.s - z15.s}  // 11000000-10000100-10100101-10000011
// CHECK-INST: mov     za3v.s[w13, 0:3], { z12.s - z15.s }
// CHECK-ENCODING: [0x83,0xa5,0x84,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c084a583 <unknown>


mova    {z0.d - z3.d}, za0h.d[w12, 0:3]  // 11000000-11000110-00000100-00000000
// CHECK-INST: mov     { z0.d - z3.d }, za0h.d[w12, 0:3]
// CHECK-ENCODING: [0x00,0x04,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c60400 <unknown>

mova    {z20.d - z23.d}, za2h.d[w14, 0:3]  // 11000000-11000110-01000100-01010100
// CHECK-INST: mov     { z20.d - z23.d }, za2h.d[w14, 0:3]
// CHECK-ENCODING: [0x54,0x44,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c64454 <unknown>

mova    {z20.d - z23.d}, za5h.d[w15, 0:3]  // 11000000-11000110-01100100-10110100
// CHECK-INST: mov     { z20.d - z23.d }, za5h.d[w15, 0:3]
// CHECK-ENCODING: [0xb4,0x64,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c664b4 <unknown>

mova    {z28.d - z31.d}, za7h.d[w15, 0:3]  // 11000000-11000110-01100100-11111100
// CHECK-INST: mov     { z28.d - z31.d }, za7h.d[w15, 0:3]
// CHECK-ENCODING: [0xfc,0x64,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c664fc <unknown>

mova    {z4.d - z7.d}, za1h.d[w12, 0:3]  // 11000000-11000110-00000100-00100100
// CHECK-INST: mov     { z4.d - z7.d }, za1h.d[w12, 0:3]
// CHECK-ENCODING: [0x24,0x04,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c60424 <unknown>

mova    {z0.d - z3.d}, za1h.d[w12, 0:3]  // 11000000-11000110-00000100-00100000
// CHECK-INST: mov     { z0.d - z3.d }, za1h.d[w12, 0:3]
// CHECK-ENCODING: [0x20,0x04,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c60420 <unknown>

mova    {z24.d - z27.d}, za3h.d[w14, 0:3]  // 11000000-11000110-01000100-01111000
// CHECK-INST: mov     { z24.d - z27.d }, za3h.d[w14, 0:3]
// CHECK-ENCODING: [0x78,0x44,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c64478 <unknown>

mova    {z0.d - z3.d}, za4h.d[w12, 0:3]  // 11000000-11000110-00000100-10000000
// CHECK-INST: mov     { z0.d - z3.d }, za4h.d[w12, 0:3]
// CHECK-ENCODING: [0x80,0x04,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c60480 <unknown>

mova    {z16.d - z19.d}, za1h.d[w14, 0:3]  // 11000000-11000110-01000100-00110000
// CHECK-INST: mov     { z16.d - z19.d }, za1h.d[w14, 0:3]
// CHECK-ENCODING: [0x30,0x44,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c64430 <unknown>

mova    {z28.d - z31.d}, za6h.d[w12, 0:3]  // 11000000-11000110-00000100-11011100
// CHECK-INST: mov     { z28.d - z31.d }, za6h.d[w12, 0:3]
// CHECK-ENCODING: [0xdc,0x04,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c604dc <unknown>

mova    {z0.d - z3.d}, za1h.d[w15, 0:3]  // 11000000-11000110-01100100-00100000
// CHECK-INST: mov     { z0.d - z3.d }, za1h.d[w15, 0:3]
// CHECK-ENCODING: [0x20,0x64,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c66420 <unknown>

mova    {z4.d - z7.d}, za4h.d[w13, 0:3]  // 11000000-11000110-00100100-10000100
// CHECK-INST: mov     { z4.d - z7.d }, za4h.d[w13, 0:3]
// CHECK-ENCODING: [0x84,0x24,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c62484 <unknown>

// Aliases

mov     {z0.d - z3.d}, za0h.d[w12, 0:3]  // 11000000-11000110-00000100-00000000
// CHECK-INST: mov     { z0.d - z3.d }, za0h.d[w12, 0:3]
// CHECK-ENCODING: [0x00,0x04,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c60400 <unknown>

mov     {z20.d - z23.d}, za2h.d[w14, 0:3]  // 11000000-11000110-01000100-01010100
// CHECK-INST: mov     { z20.d - z23.d }, za2h.d[w14, 0:3]
// CHECK-ENCODING: [0x54,0x44,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c64454 <unknown>

mov     {z20.d - z23.d}, za5h.d[w15, 0:3]  // 11000000-11000110-01100100-10110100
// CHECK-INST: mov     { z20.d - z23.d }, za5h.d[w15, 0:3]
// CHECK-ENCODING: [0xb4,0x64,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c664b4 <unknown>

mov     {z28.d - z31.d}, za7h.d[w15, 0:3]  // 11000000-11000110-01100100-11111100
// CHECK-INST: mov     { z28.d - z31.d }, za7h.d[w15, 0:3]
// CHECK-ENCODING: [0xfc,0x64,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c664fc <unknown>

mov     {z4.d - z7.d}, za1h.d[w12, 0:3]  // 11000000-11000110-00000100-00100100
// CHECK-INST: mov     { z4.d - z7.d }, za1h.d[w12, 0:3]
// CHECK-ENCODING: [0x24,0x04,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c60424 <unknown>

mov     {z0.d - z3.d}, za1h.d[w12, 0:3]  // 11000000-11000110-00000100-00100000
// CHECK-INST: mov     { z0.d - z3.d }, za1h.d[w12, 0:3]
// CHECK-ENCODING: [0x20,0x04,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c60420 <unknown>

mov     {z24.d - z27.d}, za3h.d[w14, 0:3]  // 11000000-11000110-01000100-01111000
// CHECK-INST: mov     { z24.d - z27.d }, za3h.d[w14, 0:3]
// CHECK-ENCODING: [0x78,0x44,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c64478 <unknown>

mov     {z0.d - z3.d}, za4h.d[w12, 0:3]  // 11000000-11000110-00000100-10000000
// CHECK-INST: mov     { z0.d - z3.d }, za4h.d[w12, 0:3]
// CHECK-ENCODING: [0x80,0x04,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c60480 <unknown>

mov     {z16.d - z19.d}, za1h.d[w14, 0:3]  // 11000000-11000110-01000100-00110000
// CHECK-INST: mov     { z16.d - z19.d }, za1h.d[w14, 0:3]
// CHECK-ENCODING: [0x30,0x44,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c64430 <unknown>

mov     {z28.d - z31.d}, za6h.d[w12, 0:3]  // 11000000-11000110-00000100-11011100
// CHECK-INST: mov     { z28.d - z31.d }, za6h.d[w12, 0:3]
// CHECK-ENCODING: [0xdc,0x04,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c604dc <unknown>

mov     {z0.d - z3.d}, za1h.d[w15, 0:3]  // 11000000-11000110-01100100-00100000
// CHECK-INST: mov     { z0.d - z3.d }, za1h.d[w15, 0:3]
// CHECK-ENCODING: [0x20,0x64,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c66420 <unknown>

mov     {z4.d - z7.d}, za4h.d[w13, 0:3]  // 11000000-11000110-00100100-10000100
// CHECK-INST: mov     { z4.d - z7.d }, za4h.d[w13, 0:3]
// CHECK-ENCODING: [0x84,0x24,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c62484 <unknown>


mova    {z0.d - z3.d}, za0v.d[w12, 0:3]  // 11000000-11000110-10000100-00000000
// CHECK-INST: mov     { z0.d - z3.d }, za0v.d[w12, 0:3]
// CHECK-ENCODING: [0x00,0x84,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c68400 <unknown>

mova    {z20.d - z23.d}, za2v.d[w14, 0:3]  // 11000000-11000110-11000100-01010100
// CHECK-INST: mov     { z20.d - z23.d }, za2v.d[w14, 0:3]
// CHECK-ENCODING: [0x54,0xc4,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6c454 <unknown>

mova    {z20.d - z23.d}, za5v.d[w15, 0:3]  // 11000000-11000110-11100100-10110100
// CHECK-INST: mov     { z20.d - z23.d }, za5v.d[w15, 0:3]
// CHECK-ENCODING: [0xb4,0xe4,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6e4b4 <unknown>

mova    {z28.d - z31.d}, za7v.d[w15, 0:3]  // 11000000-11000110-11100100-11111100
// CHECK-INST: mov     { z28.d - z31.d }, za7v.d[w15, 0:3]
// CHECK-ENCODING: [0xfc,0xe4,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6e4fc <unknown>

mova    {z4.d - z7.d}, za1v.d[w12, 0:3]  // 11000000-11000110-10000100-00100100
// CHECK-INST: mov     { z4.d - z7.d }, za1v.d[w12, 0:3]
// CHECK-ENCODING: [0x24,0x84,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c68424 <unknown>

mova    {z0.d - z3.d}, za1v.d[w12, 0:3]  // 11000000-11000110-10000100-00100000
// CHECK-INST: mov     { z0.d - z3.d }, za1v.d[w12, 0:3]
// CHECK-ENCODING: [0x20,0x84,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c68420 <unknown>

mova    {z24.d - z27.d}, za3v.d[w14, 0:3]  // 11000000-11000110-11000100-01111000
// CHECK-INST: mov     { z24.d - z27.d }, za3v.d[w14, 0:3]
// CHECK-ENCODING: [0x78,0xc4,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6c478 <unknown>

mova    {z0.d - z3.d}, za4v.d[w12, 0:3]  // 11000000-11000110-10000100-10000000
// CHECK-INST: mov     { z0.d - z3.d }, za4v.d[w12, 0:3]
// CHECK-ENCODING: [0x80,0x84,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c68480 <unknown>

mova    {z16.d - z19.d}, za1v.d[w14, 0:3]  // 11000000-11000110-11000100-00110000
// CHECK-INST: mov     { z16.d - z19.d }, za1v.d[w14, 0:3]
// CHECK-ENCODING: [0x30,0xc4,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6c430 <unknown>

mova    {z28.d - z31.d}, za6v.d[w12, 0:3]  // 11000000-11000110-10000100-11011100
// CHECK-INST: mov     { z28.d - z31.d }, za6v.d[w12, 0:3]
// CHECK-ENCODING: [0xdc,0x84,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c684dc <unknown>

mova    {z0.d - z3.d}, za1v.d[w15, 0:3]  // 11000000-11000110-11100100-00100000
// CHECK-INST: mov     { z0.d - z3.d }, za1v.d[w15, 0:3]
// CHECK-ENCODING: [0x20,0xe4,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6e420 <unknown>

mova    {z4.d - z7.d}, za4v.d[w13, 0:3]  // 11000000-11000110-10100100-10000100
// CHECK-INST: mov     { z4.d - z7.d }, za4v.d[w13, 0:3]
// CHECK-ENCODING: [0x84,0xa4,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6a484 <unknown>

// Aliases

mov     {z0.d - z3.d}, za0v.d[w12, 0:3]  // 11000000-11000110-10000100-00000000
// CHECK-INST: mov     { z0.d - z3.d }, za0v.d[w12, 0:3]
// CHECK-ENCODING: [0x00,0x84,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c68400 <unknown>

mov     {z20.d - z23.d}, za2v.d[w14, 0:3]  // 11000000-11000110-11000100-01010100
// CHECK-INST: mov     { z20.d - z23.d }, za2v.d[w14, 0:3]
// CHECK-ENCODING: [0x54,0xc4,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6c454 <unknown>

mov     {z20.d - z23.d}, za5v.d[w15, 0:3]  // 11000000-11000110-11100100-10110100
// CHECK-INST: mov     { z20.d - z23.d }, za5v.d[w15, 0:3]
// CHECK-ENCODING: [0xb4,0xe4,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6e4b4 <unknown>

mov     {z28.d - z31.d}, za7v.d[w15, 0:3]  // 11000000-11000110-11100100-11111100
// CHECK-INST: mov     { z28.d - z31.d }, za7v.d[w15, 0:3]
// CHECK-ENCODING: [0xfc,0xe4,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6e4fc <unknown>

mov     {z4.d - z7.d}, za1v.d[w12, 0:3]  // 11000000-11000110-10000100-00100100
// CHECK-INST: mov     { z4.d - z7.d }, za1v.d[w12, 0:3]
// CHECK-ENCODING: [0x24,0x84,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c68424 <unknown>

mov     {z0.d - z3.d}, za1v.d[w12, 0:3]  // 11000000-11000110-10000100-00100000
// CHECK-INST: mov     { z0.d - z3.d }, za1v.d[w12, 0:3]
// CHECK-ENCODING: [0x20,0x84,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c68420 <unknown>

mov     {z24.d - z27.d}, za3v.d[w14, 0:3]  // 11000000-11000110-11000100-01111000
// CHECK-INST: mov     { z24.d - z27.d }, za3v.d[w14, 0:3]
// CHECK-ENCODING: [0x78,0xc4,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6c478 <unknown>

mov     {z0.d - z3.d}, za4v.d[w12, 0:3]  // 11000000-11000110-10000100-10000000
// CHECK-INST: mov     { z0.d - z3.d }, za4v.d[w12, 0:3]
// CHECK-ENCODING: [0x80,0x84,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c68480 <unknown>

mov     {z16.d - z19.d}, za1v.d[w14, 0:3]  // 11000000-11000110-11000100-00110000
// CHECK-INST: mov     { z16.d - z19.d }, za1v.d[w14, 0:3]
// CHECK-ENCODING: [0x30,0xc4,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6c430 <unknown>

mov     {z28.d - z31.d}, za6v.d[w12, 0:3]  // 11000000-11000110-10000100-11011100
// CHECK-INST: mov     { z28.d - z31.d }, za6v.d[w12, 0:3]
// CHECK-ENCODING: [0xdc,0x84,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c684dc <unknown>

mov     {z0.d - z3.d}, za1v.d[w15, 0:3]  // 11000000-11000110-11100100-00100000
// CHECK-INST: mov     { z0.d - z3.d }, za1v.d[w15, 0:3]
// CHECK-ENCODING: [0x20,0xe4,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6e420 <unknown>

mov     {z4.d - z7.d}, za4v.d[w13, 0:3]  // 11000000-11000110-10100100-10000100
// CHECK-INST: mov     { z4.d - z7.d }, za4v.d[w13, 0:3]
// CHECK-ENCODING: [0x84,0xa4,0xc6,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c6a484 <unknown>


mova    {z0.d - z3.d}, za.d[w8, 0, vgx4]  // 11000000-00000110-00001100-00000000
// CHECK-INST: mov     { z0.d - z3.d }, za.d[w8, 0, vgx4]
// CHECK-ENCODING: [0x00,0x0c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060c00 <unknown>

mova    {z0.d - z3.d}, za.d[w8, 0]  // 11000000-00000110-00001100-00000000
// CHECK-INST: mov     { z0.d - z3.d }, za.d[w8, 0, vgx4]
// CHECK-ENCODING: [0x00,0x0c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060c00 <unknown>

mova    {z20.d - z23.d}, za.d[w10, 2, vgx4]  // 11000000-00000110-01001100-01010100
// CHECK-INST: mov     { z20.d - z23.d }, za.d[w10, 2, vgx4]
// CHECK-ENCODING: [0x54,0x4c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064c54 <unknown>

mova    {z20.d - z23.d}, za.d[w10, 2]  // 11000000-00000110-01001100-01010100
// CHECK-INST: mov     { z20.d - z23.d }, za.d[w10, 2, vgx4]
// CHECK-ENCODING: [0x54,0x4c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064c54 <unknown>

mova    {z20.d - z23.d}, za.d[w11, 5, vgx4]  // 11000000-00000110-01101100-10110100
// CHECK-INST: mov     { z20.d - z23.d }, za.d[w11, 5, vgx4]
// CHECK-ENCODING: [0xb4,0x6c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066cb4 <unknown>

mova    {z20.d - z23.d}, za.d[w11, 5]  // 11000000-00000110-01101100-10110100
// CHECK-INST: mov     { z20.d - z23.d }, za.d[w11, 5, vgx4]
// CHECK-ENCODING: [0xb4,0x6c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066cb4 <unknown>

mova    {z28.d - z31.d}, za.d[w11, 7, vgx4]  // 11000000-00000110-01101100-11111100
// CHECK-INST: mov     { z28.d - z31.d }, za.d[w11, 7, vgx4]
// CHECK-ENCODING: [0xfc,0x6c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066cfc <unknown>

mova    {z28.d - z31.d}, za.d[w11, 7]  // 11000000-00000110-01101100-11111100
// CHECK-INST: mov     { z28.d - z31.d }, za.d[w11, 7, vgx4]
// CHECK-ENCODING: [0xfc,0x6c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066cfc <unknown>

mova    {z4.d - z7.d}, za.d[w8, 1, vgx4]  // 11000000-00000110-00001100-00100100
// CHECK-INST: mov     { z4.d - z7.d }, za.d[w8, 1, vgx4]
// CHECK-ENCODING: [0x24,0x0c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060c24 <unknown>

mova    {z4.d - z7.d}, za.d[w8, 1]  // 11000000-00000110-00001100-00100100
// CHECK-INST: mov     { z4.d - z7.d }, za.d[w8, 1, vgx4]
// CHECK-ENCODING: [0x24,0x0c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060c24 <unknown>

mova    {z0.d - z3.d}, za.d[w8, 1, vgx4]  // 11000000-00000110-00001100-00100000
// CHECK-INST: mov     { z0.d - z3.d }, za.d[w8, 1, vgx4]
// CHECK-ENCODING: [0x20,0x0c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060c20 <unknown>

mova    {z0.d - z3.d}, za.d[w8, 1]  // 11000000-00000110-00001100-00100000
// CHECK-INST: mov     { z0.d - z3.d }, za.d[w8, 1, vgx4]
// CHECK-ENCODING: [0x20,0x0c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060c20 <unknown>

mova    {z24.d - z27.d}, za.d[w10, 3, vgx4]  // 11000000-00000110-01001100-01111000
// CHECK-INST: mov     { z24.d - z27.d }, za.d[w10, 3, vgx4]
// CHECK-ENCODING: [0x78,0x4c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064c78 <unknown>

mova    {z24.d - z27.d}, za.d[w10, 3]  // 11000000-00000110-01001100-01111000
// CHECK-INST: mov     { z24.d - z27.d }, za.d[w10, 3, vgx4]
// CHECK-ENCODING: [0x78,0x4c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064c78 <unknown>

mova    {z0.d - z3.d}, za.d[w8, 4, vgx4]  // 11000000-00000110-00001100-10000000
// CHECK-INST: mov     { z0.d - z3.d }, za.d[w8, 4, vgx4]
// CHECK-ENCODING: [0x80,0x0c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060c80 <unknown>

mova    {z0.d - z3.d}, za.d[w8, 4]  // 11000000-00000110-00001100-10000000
// CHECK-INST: mov     { z0.d - z3.d }, za.d[w8, 4, vgx4]
// CHECK-ENCODING: [0x80,0x0c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060c80 <unknown>

mova    {z16.d - z19.d}, za.d[w10, 1, vgx4]  // 11000000-00000110-01001100-00110000
// CHECK-INST: mov     { z16.d - z19.d }, za.d[w10, 1, vgx4]
// CHECK-ENCODING: [0x30,0x4c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064c30 <unknown>

mova    {z16.d - z19.d}, za.d[w10, 1]  // 11000000-00000110-01001100-00110000
// CHECK-INST: mov     { z16.d - z19.d }, za.d[w10, 1, vgx4]
// CHECK-ENCODING: [0x30,0x4c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064c30 <unknown>

mova    {z28.d - z31.d}, za.d[w8, 6, vgx4]  // 11000000-00000110-00001100-11011100
// CHECK-INST: mov     { z28.d - z31.d }, za.d[w8, 6, vgx4]
// CHECK-ENCODING: [0xdc,0x0c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060cdc <unknown>

mova    {z28.d - z31.d}, za.d[w8, 6]  // 11000000-00000110-00001100-11011100
// CHECK-INST: mov     { z28.d - z31.d }, za.d[w8, 6, vgx4]
// CHECK-ENCODING: [0xdc,0x0c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060cdc <unknown>

mova    {z0.d - z3.d}, za.d[w11, 1, vgx4]  // 11000000-00000110-01101100-00100000
// CHECK-INST: mov     { z0.d - z3.d }, za.d[w11, 1, vgx4]
// CHECK-ENCODING: [0x20,0x6c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066c20 <unknown>

mova    {z0.d - z3.d}, za.d[w11, 1]  // 11000000-00000110-01101100-00100000
// CHECK-INST: mov     { z0.d - z3.d }, za.d[w11, 1, vgx4]
// CHECK-ENCODING: [0x20,0x6c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066c20 <unknown>

mova    {z4.d - z7.d}, za.d[w9, 4, vgx4]  // 11000000-00000110-00101100-10000100
// CHECK-INST: mov     { z4.d - z7.d }, za.d[w9, 4, vgx4]
// CHECK-ENCODING: [0x84,0x2c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0062c84 <unknown>

mova    {z4.d - z7.d}, za.d[w9, 4]  // 11000000-00000110-00101100-10000100
// CHECK-INST: mov     { z4.d - z7.d }, za.d[w9, 4, vgx4]
// CHECK-ENCODING: [0x84,0x2c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0062c84 <unknown>

// Aliases

mov     {z0.d - z3.d}, za.d[w8, 0, vgx4]  // 11000000-00000110-00001100-00000000
// CHECK-INST: mov     { z0.d - z3.d }, za.d[w8, 0, vgx4]
// CHECK-ENCODING: [0x00,0x0c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060c00 <unknown>

mov     {z20.d - z23.d}, za.d[w10, 2, vgx4]  // 11000000-00000110-01001100-01010100
// CHECK-INST: mov     { z20.d - z23.d }, za.d[w10, 2, vgx4]
// CHECK-ENCODING: [0x54,0x4c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064c54 <unknown>

mov     {z20.d - z23.d}, za.d[w11, 5, vgx4]  // 11000000-00000110-01101100-10110100
// CHECK-INST: mov     { z20.d - z23.d }, za.d[w11, 5, vgx4]
// CHECK-ENCODING: [0xb4,0x6c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066cb4 <unknown>

mov     {z28.d - z31.d}, za.d[w11, 7, vgx4]  // 11000000-00000110-01101100-11111100
// CHECK-INST: mov     { z28.d - z31.d }, za.d[w11, 7, vgx4]
// CHECK-ENCODING: [0xfc,0x6c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066cfc <unknown>

mov     {z4.d - z7.d}, za.d[w8, 1, vgx4]  // 11000000-00000110-00001100-00100100
// CHECK-INST: mov     { z4.d - z7.d }, za.d[w8, 1, vgx4]
// CHECK-ENCODING: [0x24,0x0c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060c24 <unknown>

mov     {z0.d - z3.d}, za.d[w8, 1, vgx4]  // 11000000-00000110-00001100-00100000
// CHECK-INST: mov     { z0.d - z3.d }, za.d[w8, 1, vgx4]
// CHECK-ENCODING: [0x20,0x0c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060c20 <unknown>

mov     {z24.d - z27.d}, za.d[w10, 3, vgx4]  // 11000000-00000110-01001100-01111000
// CHECK-INST: mov     { z24.d - z27.d }, za.d[w10, 3, vgx4]
// CHECK-ENCODING: [0x78,0x4c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064c78 <unknown>

mov     {z0.d - z3.d}, za.d[w8, 4, vgx4]  // 11000000-00000110-00001100-10000000
// CHECK-INST: mov     { z0.d - z3.d }, za.d[w8, 4, vgx4]
// CHECK-ENCODING: [0x80,0x0c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060c80 <unknown>

mov     {z16.d - z19.d}, za.d[w10, 1, vgx4]  // 11000000-00000110-01001100-00110000
// CHECK-INST: mov     { z16.d - z19.d }, za.d[w10, 1, vgx4]
// CHECK-ENCODING: [0x30,0x4c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064c30 <unknown>

mov     {z28.d - z31.d}, za.d[w8, 6, vgx4]  // 11000000-00000110-00001100-11011100
// CHECK-INST: mov     { z28.d - z31.d }, za.d[w8, 6, vgx4]
// CHECK-ENCODING: [0xdc,0x0c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060cdc <unknown>

mov     {z0.d - z3.d}, za.d[w11, 1, vgx4]  // 11000000-00000110-01101100-00100000
// CHECK-INST: mov     { z0.d - z3.d }, za.d[w11, 1, vgx4]
// CHECK-ENCODING: [0x20,0x6c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066c20 <unknown>

mov     {z4.d - z7.d}, za.d[w9, 4, vgx4]  // 11000000-00000110-00101100-10000100
// CHECK-INST: mov     { z4.d - z7.d }, za.d[w9, 4, vgx4]
// CHECK-ENCODING: [0x84,0x2c,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0062c84 <unknown>


mova    za0h.d[w12, 0:3], {z0.d - z3.d}  // 11000000-11000100-00000100-00000000
// CHECK-INST: mov     za0h.d[w12, 0:3], { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0x04,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40400 <unknown>

mova    za5h.d[w14, 0:3], {z8.d - z11.d}  // 11000000-11000100-01000101-00000101
// CHECK-INST: mov     za5h.d[w14, 0:3], { z8.d - z11.d }
// CHECK-ENCODING: [0x05,0x45,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c44505 <unknown>

mova    za7h.d[w15, 0:3], {z12.d - z15.d}  // 11000000-11000100-01100101-10000111
// CHECK-INST: mov     za7h.d[w15, 0:3], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0x65,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c46587 <unknown>

mova    za7h.d[w15, 0:3], {z28.d - z31.d}  // 11000000-11000100-01100111-10000111
// CHECK-INST: mov     za7h.d[w15, 0:3], { z28.d - z31.d }
// CHECK-ENCODING: [0x87,0x67,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c46787 <unknown>

mova    za5h.d[w12, 0:3], {z16.d - z19.d}  // 11000000-11000100-00000110-00000101
// CHECK-INST: mov     za5h.d[w12, 0:3], { z16.d - z19.d }
// CHECK-ENCODING: [0x05,0x06,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40605 <unknown>

mova    za1h.d[w12, 0:3], {z0.d - z3.d}  // 11000000-11000100-00000100-00000001
// CHECK-INST: mov     za1h.d[w12, 0:3], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0x04,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40401 <unknown>

mova    za0h.d[w14, 0:3], {z16.d - z19.d}  // 11000000-11000100-01000110-00000000
// CHECK-INST: mov     za0h.d[w14, 0:3], { z16.d - z19.d }
// CHECK-ENCODING: [0x00,0x46,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c44600 <unknown>

mova    za0h.d[w12, 0:3], {z12.d - z15.d}  // 11000000-11000100-00000101-10000000
// CHECK-INST: mov     za0h.d[w12, 0:3], { z12.d - z15.d }
// CHECK-ENCODING: [0x80,0x05,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40580 <unknown>

mova    za1h.d[w14, 0:3], {z0.d - z3.d}  // 11000000-11000100-01000100-00000001
// CHECK-INST: mov     za1h.d[w14, 0:3], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0x44,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c44401 <unknown>

mova    za5h.d[w12, 0:3], {z20.d - z23.d}  // 11000000-11000100-00000110-10000101
// CHECK-INST: mov     za5h.d[w12, 0:3], { z20.d - z23.d }
// CHECK-ENCODING: [0x85,0x06,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40685 <unknown>

mova    za2h.d[w15, 0:3], {z8.d - z11.d}  // 11000000-11000100-01100101-00000010
// CHECK-INST: mov     za2h.d[w15, 0:3], { z8.d - z11.d }
// CHECK-ENCODING: [0x02,0x65,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c46502 <unknown>

mova    za7h.d[w13, 0:3], {z12.d - z15.d}  // 11000000-11000100-00100101-10000111
// CHECK-INST: mov     za7h.d[w13, 0:3], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0x25,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c42587 <unknown>

// Aliases

mov     za0h.d[w12, 0:3], {z0.d - z3.d}  // 11000000-11000100-00000100-00000000
// CHECK-INST: mov     za0h.d[w12, 0:3], { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0x04,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40400 <unknown>

mov     za5h.d[w14, 0:3], {z8.d - z11.d}  // 11000000-11000100-01000101-00000101
// CHECK-INST: mov     za5h.d[w14, 0:3], { z8.d - z11.d }
// CHECK-ENCODING: [0x05,0x45,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c44505 <unknown>

mov     za7h.d[w15, 0:3], {z12.d - z15.d}  // 11000000-11000100-01100101-10000111
// CHECK-INST: mov     za7h.d[w15, 0:3], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0x65,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c46587 <unknown>

mov     za7h.d[w15, 0:3], {z28.d - z31.d}  // 11000000-11000100-01100111-10000111
// CHECK-INST: mov     za7h.d[w15, 0:3], { z28.d - z31.d }
// CHECK-ENCODING: [0x87,0x67,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c46787 <unknown>

mov     za5h.d[w12, 0:3], {z16.d - z19.d}  // 11000000-11000100-00000110-00000101
// CHECK-INST: mov     za5h.d[w12, 0:3], { z16.d - z19.d }
// CHECK-ENCODING: [0x05,0x06,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40605 <unknown>

mov     za1h.d[w12, 0:3], {z0.d - z3.d}  // 11000000-11000100-00000100-00000001
// CHECK-INST: mov     za1h.d[w12, 0:3], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0x04,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40401 <unknown>

mov     za0h.d[w14, 0:3], {z16.d - z19.d}  // 11000000-11000100-01000110-00000000
// CHECK-INST: mov     za0h.d[w14, 0:3], { z16.d - z19.d }
// CHECK-ENCODING: [0x00,0x46,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c44600 <unknown>

mov     za0h.d[w12, 0:3], {z12.d - z15.d}  // 11000000-11000100-00000101-10000000
// CHECK-INST: mov     za0h.d[w12, 0:3], { z12.d - z15.d }
// CHECK-ENCODING: [0x80,0x05,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40580 <unknown>

mov     za1h.d[w14, 0:3], {z0.d - z3.d}  // 11000000-11000100-01000100-00000001
// CHECK-INST: mov     za1h.d[w14, 0:3], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0x44,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c44401 <unknown>

mov     za5h.d[w12, 0:3], {z20.d - z23.d}  // 11000000-11000100-00000110-10000101
// CHECK-INST: mov     za5h.d[w12, 0:3], { z20.d - z23.d }
// CHECK-ENCODING: [0x85,0x06,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c40685 <unknown>

mov     za2h.d[w15, 0:3], {z8.d - z11.d}  // 11000000-11000100-01100101-00000010
// CHECK-INST: mov     za2h.d[w15, 0:3], { z8.d - z11.d }
// CHECK-ENCODING: [0x02,0x65,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c46502 <unknown>

mov     za7h.d[w13, 0:3], {z12.d - z15.d}  // 11000000-11000100-00100101-10000111
// CHECK-INST: mov     za7h.d[w13, 0:3], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0x25,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c42587 <unknown>


mova    za0v.d[w12, 0:3], {z0.d - z3.d}  // 11000000-11000100-10000100-00000000
// CHECK-INST: mov     za0v.d[w12, 0:3], { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0x84,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48400 <unknown>

mova    za5v.d[w14, 0:3], {z8.d - z11.d}  // 11000000-11000100-11000101-00000101
// CHECK-INST: mov     za5v.d[w14, 0:3], { z8.d - z11.d }
// CHECK-ENCODING: [0x05,0xc5,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4c505 <unknown>

mova    za7v.d[w15, 0:3], {z12.d - z15.d}  // 11000000-11000100-11100101-10000111
// CHECK-INST: mov     za7v.d[w15, 0:3], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0xe5,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4e587 <unknown>

mova    za7v.d[w15, 0:3], {z28.d - z31.d}  // 11000000-11000100-11100111-10000111
// CHECK-INST: mov     za7v.d[w15, 0:3], { z28.d - z31.d }
// CHECK-ENCODING: [0x87,0xe7,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4e787 <unknown>

mova    za5v.d[w12, 0:3], {z16.d - z19.d}  // 11000000-11000100-10000110-00000101
// CHECK-INST: mov     za5v.d[w12, 0:3], { z16.d - z19.d }
// CHECK-ENCODING: [0x05,0x86,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48605 <unknown>

mova    za1v.d[w12, 0:3], {z0.d - z3.d}  // 11000000-11000100-10000100-00000001
// CHECK-INST: mov     za1v.d[w12, 0:3], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0x84,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48401 <unknown>

mova    za0v.d[w14, 0:3], {z16.d - z19.d}  // 11000000-11000100-11000110-00000000
// CHECK-INST: mov     za0v.d[w14, 0:3], { z16.d - z19.d }
// CHECK-ENCODING: [0x00,0xc6,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4c600 <unknown>

mova    za0v.d[w12, 0:3], {z12.d - z15.d}  // 11000000-11000100-10000101-10000000
// CHECK-INST: mov     za0v.d[w12, 0:3], { z12.d - z15.d }
// CHECK-ENCODING: [0x80,0x85,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48580 <unknown>

mova    za1v.d[w14, 0:3], {z0.d - z3.d}  // 11000000-11000100-11000100-00000001
// CHECK-INST: mov     za1v.d[w14, 0:3], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0xc4,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4c401 <unknown>

mova    za5v.d[w12, 0:3], {z20.d - z23.d}  // 11000000-11000100-10000110-10000101
// CHECK-INST: mov     za5v.d[w12, 0:3], { z20.d - z23.d }
// CHECK-ENCODING: [0x85,0x86,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48685 <unknown>

mova    za2v.d[w15, 0:3], {z8.d - z11.d}  // 11000000-11000100-11100101-00000010
// CHECK-INST: mov     za2v.d[w15, 0:3], { z8.d - z11.d }
// CHECK-ENCODING: [0x02,0xe5,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4e502 <unknown>

mova    za7v.d[w13, 0:3], {z12.d - z15.d}  // 11000000-11000100-10100101-10000111
// CHECK-INST: mov     za7v.d[w13, 0:3], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0xa5,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4a587 <unknown>

// Aliases

mov     za0v.d[w12, 0:3], {z0.d - z3.d}  // 11000000-11000100-10000100-00000000
// CHECK-INST: mov     za0v.d[w12, 0:3], { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0x84,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48400 <unknown>

mov     za5v.d[w14, 0:3], {z8.d - z11.d}  // 11000000-11000100-11000101-00000101
// CHECK-INST: mov     za5v.d[w14, 0:3], { z8.d - z11.d }
// CHECK-ENCODING: [0x05,0xc5,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4c505 <unknown>

mov     za7v.d[w15, 0:3], {z12.d - z15.d}  // 11000000-11000100-11100101-10000111
// CHECK-INST: mov     za7v.d[w15, 0:3], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0xe5,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4e587 <unknown>

mov     za7v.d[w15, 0:3], {z28.d - z31.d}  // 11000000-11000100-11100111-10000111
// CHECK-INST: mov     za7v.d[w15, 0:3], { z28.d - z31.d }
// CHECK-ENCODING: [0x87,0xe7,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4e787 <unknown>

mov     za5v.d[w12, 0:3], {z16.d - z19.d}  // 11000000-11000100-10000110-00000101
// CHECK-INST: mov     za5v.d[w12, 0:3], { z16.d - z19.d }
// CHECK-ENCODING: [0x05,0x86,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48605 <unknown>

mov     za1v.d[w12, 0:3], {z0.d - z3.d}  // 11000000-11000100-10000100-00000001
// CHECK-INST: mov     za1v.d[w12, 0:3], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0x84,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48401 <unknown>

mov     za0v.d[w14, 0:3], {z16.d - z19.d}  // 11000000-11000100-11000110-00000000
// CHECK-INST: mov     za0v.d[w14, 0:3], { z16.d - z19.d }
// CHECK-ENCODING: [0x00,0xc6,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4c600 <unknown>

mov     za0v.d[w12, 0:3], {z12.d - z15.d}  // 11000000-11000100-10000101-10000000
// CHECK-INST: mov     za0v.d[w12, 0:3], { z12.d - z15.d }
// CHECK-ENCODING: [0x80,0x85,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48580 <unknown>

mov     za1v.d[w14, 0:3], {z0.d - z3.d}  // 11000000-11000100-11000100-00000001
// CHECK-INST: mov     za1v.d[w14, 0:3], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0xc4,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4c401 <unknown>

mov     za5v.d[w12, 0:3], {z20.d - z23.d}  // 11000000-11000100-10000110-10000101
// CHECK-INST: mov     za5v.d[w12, 0:3], { z20.d - z23.d }
// CHECK-ENCODING: [0x85,0x86,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c48685 <unknown>

mov     za2v.d[w15, 0:3], {z8.d - z11.d}  // 11000000-11000100-11100101-00000010
// CHECK-INST: mov     za2v.d[w15, 0:3], { z8.d - z11.d }
// CHECK-ENCODING: [0x02,0xe5,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4e502 <unknown>

mov     za7v.d[w13, 0:3], {z12.d - z15.d}  // 11000000-11000100-10100101-10000111
// CHECK-INST: mov     za7v.d[w13, 0:3], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0xa5,0xc4,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0c4a587 <unknown>


mova    za.d[w8, 0, vgx4], {z0.d - z3.d}  // 11000000-00000100-00001100-00000000
// CHECK-INST: mov     za.d[w8, 0, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0x0c,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040c00 <unknown>

mova    za.d[w8, 0], {z0.d - z3.d}  // 11000000-00000100-00001100-00000000
// CHECK-INST: mov     za.d[w8, 0, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0x0c,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040c00 <unknown>

mova    za.d[w10, 5, vgx4], {z8.d - z11.d}  // 11000000-00000100-01001101-00000101
// CHECK-INST: mov     za.d[w10, 5, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x05,0x4d,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044d05 <unknown>

mova    za.d[w10, 5], {z8.d - z11.d}  // 11000000-00000100-01001101-00000101
// CHECK-INST: mov     za.d[w10, 5, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x05,0x4d,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044d05 <unknown>

mova    za.d[w11, 7, vgx4], {z12.d - z15.d}  // 11000000-00000100-01101101-10000111
// CHECK-INST: mov     za.d[w11, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0x6d,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046d87 <unknown>

mova    za.d[w11, 7], {z12.d - z15.d}  // 11000000-00000100-01101101-10000111
// CHECK-INST: mov     za.d[w11, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0x6d,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046d87 <unknown>

mova    za.d[w11, 7, vgx4], {z28.d - z31.d}  // 11000000-00000100-01101111-10000111
// CHECK-INST: mov     za.d[w11, 7, vgx4], { z28.d - z31.d }
// CHECK-ENCODING: [0x87,0x6f,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046f87 <unknown>

mova    za.d[w11, 7], {z28.d - z31.d}  // 11000000-00000100-01101111-10000111
// CHECK-INST: mov     za.d[w11, 7, vgx4], { z28.d - z31.d }
// CHECK-ENCODING: [0x87,0x6f,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046f87 <unknown>

mova    za.d[w8, 5, vgx4], {z16.d - z19.d}  // 11000000-00000100-00001110-00000101
// CHECK-INST: mov     za.d[w8, 5, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x05,0x0e,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040e05 <unknown>

mova    za.d[w8, 5], {z16.d - z19.d}  // 11000000-00000100-00001110-00000101
// CHECK-INST: mov     za.d[w8, 5, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x05,0x0e,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040e05 <unknown>

mova    za.d[w8, 1, vgx4], {z0.d - z3.d}  // 11000000-00000100-00001100-00000001
// CHECK-INST: mov     za.d[w8, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0x0c,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040c01 <unknown>

mova    za.d[w8, 1], {z0.d - z3.d}  // 11000000-00000100-00001100-00000001
// CHECK-INST: mov     za.d[w8, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0x0c,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040c01 <unknown>

mova    za.d[w10, 0, vgx4], {z16.d - z19.d}  // 11000000-00000100-01001110-00000000
// CHECK-INST: mov     za.d[w10, 0, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x00,0x4e,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044e00 <unknown>

mova    za.d[w10, 0], {z16.d - z19.d}  // 11000000-00000100-01001110-00000000
// CHECK-INST: mov     za.d[w10, 0, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x00,0x4e,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044e00 <unknown>

mova    za.d[w8, 0, vgx4], {z12.d - z15.d}  // 11000000-00000100-00001101-10000000
// CHECK-INST: mov     za.d[w8, 0, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x80,0x0d,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040d80 <unknown>

mova    za.d[w8, 0], {z12.d - z15.d}  // 11000000-00000100-00001101-10000000
// CHECK-INST: mov     za.d[w8, 0, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x80,0x0d,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040d80 <unknown>

mova    za.d[w10, 1, vgx4], {z0.d - z3.d}  // 11000000-00000100-01001100-00000001
// CHECK-INST: mov     za.d[w10, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0x4c,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044c01 <unknown>

mova    za.d[w10, 1], {z0.d - z3.d}  // 11000000-00000100-01001100-00000001
// CHECK-INST: mov     za.d[w10, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0x4c,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044c01 <unknown>

mova    za.d[w8, 5, vgx4], {z20.d - z23.d}  // 11000000-00000100-00001110-10000101
// CHECK-INST: mov     za.d[w8, 5, vgx4], { z20.d - z23.d }
// CHECK-ENCODING: [0x85,0x0e,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040e85 <unknown>

mova    za.d[w8, 5], {z20.d - z23.d}  // 11000000-00000100-00001110-10000101
// CHECK-INST: mov     za.d[w8, 5, vgx4], { z20.d - z23.d }
// CHECK-ENCODING: [0x85,0x0e,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040e85 <unknown>

mova    za.d[w11, 2, vgx4], {z8.d - z11.d}  // 11000000-00000100-01101101-00000010
// CHECK-INST: mov     za.d[w11, 2, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x02,0x6d,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046d02 <unknown>

mova    za.d[w11, 2], {z8.d - z11.d}  // 11000000-00000100-01101101-00000010
// CHECK-INST: mov     za.d[w11, 2, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x02,0x6d,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046d02 <unknown>

mova    za.d[w9, 7, vgx4], {z12.d - z15.d}  // 11000000-00000100-00101101-10000111
// CHECK-INST: mov     za.d[w9, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0x2d,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0042d87 <unknown>

mova    za.d[w9, 7], {z12.d - z15.d}  // 11000000-00000100-00101101-10000111
// CHECK-INST: mov     za.d[w9, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0x2d,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0042d87 <unknown>

// Aliases

mov     za.d[w8, 0, vgx4], {z0.d - z3.d}  // 11000000-00000100-00001100-00000000
// CHECK-INST: mov     za.d[w8, 0, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x00,0x0c,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040c00 <unknown>

mov     za.d[w10, 5, vgx4], {z8.d - z11.d}  // 11000000-00000100-01001101-00000101
// CHECK-INST: mov     za.d[w10, 5, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x05,0x4d,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044d05 <unknown>

mov     za.d[w11, 7, vgx4], {z12.d - z15.d}  // 11000000-00000100-01101101-10000111
// CHECK-INST: mov     za.d[w11, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0x6d,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046d87 <unknown>

mov     za.d[w11, 7, vgx4], {z28.d - z31.d}  // 11000000-00000100-01101111-10000111
// CHECK-INST: mov     za.d[w11, 7, vgx4], { z28.d - z31.d }
// CHECK-ENCODING: [0x87,0x6f,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046f87 <unknown>

mov     za.d[w8, 5, vgx4], {z16.d - z19.d}  // 11000000-00000100-00001110-00000101
// CHECK-INST: mov     za.d[w8, 5, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x05,0x0e,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040e05 <unknown>

mov     za.d[w8, 1, vgx4], {z0.d - z3.d}  // 11000000-00000100-00001100-00000001
// CHECK-INST: mov     za.d[w8, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0x0c,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040c01 <unknown>

mov     za.d[w10, 0, vgx4], {z16.d - z19.d}  // 11000000-00000100-01001110-00000000
// CHECK-INST: mov     za.d[w10, 0, vgx4], { z16.d - z19.d }
// CHECK-ENCODING: [0x00,0x4e,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044e00 <unknown>

mov     za.d[w8, 0, vgx4], {z12.d - z15.d}  // 11000000-00000100-00001101-10000000
// CHECK-INST: mov     za.d[w8, 0, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x80,0x0d,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040d80 <unknown>

mov     za.d[w10, 1, vgx4], {z0.d - z3.d}  // 11000000-00000100-01001100-00000001
// CHECK-INST: mov     za.d[w10, 1, vgx4], { z0.d - z3.d }
// CHECK-ENCODING: [0x01,0x4c,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044c01 <unknown>

mov     za.d[w8, 5, vgx4], {z20.d - z23.d}  // 11000000-00000100-00001110-10000101
// CHECK-INST: mov     za.d[w8, 5, vgx4], { z20.d - z23.d }
// CHECK-ENCODING: [0x85,0x0e,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040e85 <unknown>

mov     za.d[w11, 2, vgx4], {z8.d - z11.d}  // 11000000-00000100-01101101-00000010
// CHECK-INST: mov     za.d[w11, 2, vgx4], { z8.d - z11.d }
// CHECK-ENCODING: [0x02,0x6d,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046d02 <unknown>

mov     za.d[w9, 7, vgx4], {z12.d - z15.d}  // 11000000-00000100-00101101-10000111
// CHECK-INST: mov     za.d[w9, 7, vgx4], { z12.d - z15.d }
// CHECK-ENCODING: [0x87,0x2d,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0042d87 <unknown>


mova    {z0.b - z3.b}, za0h.b[w12, 0:3]  // 11000000-00000110-00000100-00000000
// CHECK-INST: mov     { z0.b - z3.b }, za0h.b[w12, 0:3]
// CHECK-ENCODING: [0x00,0x04,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060400 <unknown>

mova    {z20.b - z23.b}, za0h.b[w14, 8:11]  // 11000000-00000110-01000100-01010100
// CHECK-INST: mov     { z20.b - z23.b }, za0h.b[w14, 8:11]
// CHECK-ENCODING: [0x54,0x44,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064454 <unknown>

mova    {z20.b - z23.b}, za0h.b[w15, 4:7]  // 11000000-00000110-01100100-00110100
// CHECK-INST: mov     { z20.b - z23.b }, za0h.b[w15, 4:7]
// CHECK-ENCODING: [0x34,0x64,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066434 <unknown>

mova    {z28.b - z31.b}, za0h.b[w15, 12:15]  // 11000000-00000110-01100100-01111100
// CHECK-INST: mov     { z28.b - z31.b }, za0h.b[w15, 12:15]
// CHECK-ENCODING: [0x7c,0x64,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006647c <unknown>

mova    {z4.b - z7.b}, za0h.b[w12, 4:7]  // 11000000-00000110-00000100-00100100
// CHECK-INST: mov     { z4.b - z7.b }, za0h.b[w12, 4:7]
// CHECK-ENCODING: [0x24,0x04,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060424 <unknown>

mova    {z0.b - z3.b}, za0h.b[w12, 4:7]  // 11000000-00000110-00000100-00100000
// CHECK-INST: mov     { z0.b - z3.b }, za0h.b[w12, 4:7]
// CHECK-ENCODING: [0x20,0x04,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060420 <unknown>

mova    {z24.b - z27.b}, za0h.b[w14, 12:15]  // 11000000-00000110-01000100-01111000
// CHECK-INST: mov     { z24.b - z27.b }, za0h.b[w14, 12:15]
// CHECK-ENCODING: [0x78,0x44,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064478 <unknown>

mova    {z16.b - z19.b}, za0h.b[w14, 4:7]  // 11000000-00000110-01000100-00110000
// CHECK-INST: mov     { z16.b - z19.b }, za0h.b[w14, 4:7]
// CHECK-ENCODING: [0x30,0x44,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064430 <unknown>

mova    {z28.b - z31.b}, za0h.b[w12, 8:11]  // 11000000-00000110-00000100-01011100
// CHECK-INST: mov     { z28.b - z31.b }, za0h.b[w12, 8:11]
// CHECK-ENCODING: [0x5c,0x04,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006045c <unknown>

mova    {z0.b - z3.b}, za0h.b[w15, 4:7]  // 11000000-00000110-01100100-00100000
// CHECK-INST: mov     { z0.b - z3.b }, za0h.b[w15, 4:7]
// CHECK-ENCODING: [0x20,0x64,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066420 <unknown>

mova    {z4.b - z7.b}, za0h.b[w13, 0:3]  // 11000000-00000110-00100100-00000100
// CHECK-INST: mov     { z4.b - z7.b }, za0h.b[w13, 0:3]
// CHECK-ENCODING: [0x04,0x24,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0062404 <unknown>

// Aliases

mov     {z0.b - z3.b}, za0h.b[w12, 0:3]  // 11000000-00000110-00000100-00000000
// CHECK-INST: mov     { z0.b - z3.b }, za0h.b[w12, 0:3]
// CHECK-ENCODING: [0x00,0x04,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060400 <unknown>

mov     {z20.b - z23.b}, za0h.b[w14, 8:11]  // 11000000-00000110-01000100-01010100
// CHECK-INST: mov     { z20.b - z23.b }, za0h.b[w14, 8:11]
// CHECK-ENCODING: [0x54,0x44,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064454 <unknown>

mov     {z20.b - z23.b}, za0h.b[w15, 4:7]  // 11000000-00000110-01100100-00110100
// CHECK-INST: mov     { z20.b - z23.b }, za0h.b[w15, 4:7]
// CHECK-ENCODING: [0x34,0x64,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066434 <unknown>

mov     {z28.b - z31.b}, za0h.b[w15, 12:15]  // 11000000-00000110-01100100-01111100
// CHECK-INST: mov     { z28.b - z31.b }, za0h.b[w15, 12:15]
// CHECK-ENCODING: [0x7c,0x64,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006647c <unknown>

mov     {z4.b - z7.b}, za0h.b[w12, 4:7]  // 11000000-00000110-00000100-00100100
// CHECK-INST: mov     { z4.b - z7.b }, za0h.b[w12, 4:7]
// CHECK-ENCODING: [0x24,0x04,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060424 <unknown>

mov     {z0.b - z3.b}, za0h.b[w12, 4:7]  // 11000000-00000110-00000100-00100000
// CHECK-INST: mov     { z0.b - z3.b }, za0h.b[w12, 4:7]
// CHECK-ENCODING: [0x20,0x04,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0060420 <unknown>

mov     {z24.b - z27.b}, za0h.b[w14, 12:15]  // 11000000-00000110-01000100-01111000
// CHECK-INST: mov     { z24.b - z27.b }, za0h.b[w14, 12:15]
// CHECK-ENCODING: [0x78,0x44,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064478 <unknown>

mov     {z16.b - z19.b}, za0h.b[w14, 4:7]  // 11000000-00000110-01000100-00110000
// CHECK-INST: mov     { z16.b - z19.b }, za0h.b[w14, 4:7]
// CHECK-ENCODING: [0x30,0x44,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0064430 <unknown>

mov     {z28.b - z31.b}, za0h.b[w12, 8:11]  // 11000000-00000110-00000100-01011100
// CHECK-INST: mov     { z28.b - z31.b }, za0h.b[w12, 8:11]
// CHECK-ENCODING: [0x5c,0x04,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006045c <unknown>

mov     {z0.b - z3.b}, za0h.b[w15, 4:7]  // 11000000-00000110-01100100-00100000
// CHECK-INST: mov     { z0.b - z3.b }, za0h.b[w15, 4:7]
// CHECK-ENCODING: [0x20,0x64,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0066420 <unknown>

mov     {z4.b - z7.b}, za0h.b[w13, 0:3]  // 11000000-00000110-00100100-00000100
// CHECK-INST: mov     { z4.b - z7.b }, za0h.b[w13, 0:3]
// CHECK-ENCODING: [0x04,0x24,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0062404 <unknown>


mova    {z0.b - z3.b}, za0v.b[w12, 0:3]  // 11000000-00000110-10000100-00000000
// CHECK-INST: mov     { z0.b - z3.b }, za0v.b[w12, 0:3]
// CHECK-ENCODING: [0x00,0x84,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0068400 <unknown>

mova    {z20.b - z23.b}, za0v.b[w14, 8:11]  // 11000000-00000110-11000100-01010100
// CHECK-INST: mov     { z20.b - z23.b }, za0v.b[w14, 8:11]
// CHECK-ENCODING: [0x54,0xc4,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006c454 <unknown>

mova    {z20.b - z23.b}, za0v.b[w15, 4:7]  // 11000000-00000110-11100100-00110100
// CHECK-INST: mov     { z20.b - z23.b }, za0v.b[w15, 4:7]
// CHECK-ENCODING: [0x34,0xe4,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006e434 <unknown>

mova    {z28.b - z31.b}, za0v.b[w15, 12:15]  // 11000000-00000110-11100100-01111100
// CHECK-INST: mov     { z28.b - z31.b }, za0v.b[w15, 12:15]
// CHECK-ENCODING: [0x7c,0xe4,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006e47c <unknown>

mova    {z4.b - z7.b}, za0v.b[w12, 4:7]  // 11000000-00000110-10000100-00100100
// CHECK-INST: mov     { z4.b - z7.b }, za0v.b[w12, 4:7]
// CHECK-ENCODING: [0x24,0x84,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0068424 <unknown>

mova    {z0.b - z3.b}, za0v.b[w12, 4:7]  // 11000000-00000110-10000100-00100000
// CHECK-INST: mov     { z0.b - z3.b }, za0v.b[w12, 4:7]
// CHECK-ENCODING: [0x20,0x84,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0068420 <unknown>

mova    {z24.b - z27.b}, za0v.b[w14, 12:15]  // 11000000-00000110-11000100-01111000
// CHECK-INST: mov     { z24.b - z27.b }, za0v.b[w14, 12:15]
// CHECK-ENCODING: [0x78,0xc4,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006c478 <unknown>

mova    {z16.b - z19.b}, za0v.b[w14, 4:7]  // 11000000-00000110-11000100-00110000
// CHECK-INST: mov     { z16.b - z19.b }, za0v.b[w14, 4:7]
// CHECK-ENCODING: [0x30,0xc4,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006c430 <unknown>

mova    {z28.b - z31.b}, za0v.b[w12, 8:11]  // 11000000-00000110-10000100-01011100
// CHECK-INST: mov     { z28.b - z31.b }, za0v.b[w12, 8:11]
// CHECK-ENCODING: [0x5c,0x84,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006845c <unknown>

mova    {z0.b - z3.b}, za0v.b[w15, 4:7]  // 11000000-00000110-11100100-00100000
// CHECK-INST: mov     { z0.b - z3.b }, za0v.b[w15, 4:7]
// CHECK-ENCODING: [0x20,0xe4,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006e420 <unknown>

mova    {z4.b - z7.b}, za0v.b[w13, 0:3]  // 11000000-00000110-10100100-00000100
// CHECK-INST: mov     { z4.b - z7.b }, za0v.b[w13, 0:3]
// CHECK-ENCODING: [0x04,0xa4,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006a404 <unknown>

// Aliases

mov     {z0.b - z3.b}, za0v.b[w12, 0:3]  // 11000000-00000110-10000100-00000000
// CHECK-INST: mov     { z0.b - z3.b }, za0v.b[w12, 0:3]
// CHECK-ENCODING: [0x00,0x84,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0068400 <unknown>

mov     {z20.b - z23.b}, za0v.b[w14, 8:11]  // 11000000-00000110-11000100-01010100
// CHECK-INST: mov     { z20.b - z23.b }, za0v.b[w14, 8:11]
// CHECK-ENCODING: [0x54,0xc4,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006c454 <unknown>

mov     {z20.b - z23.b}, za0v.b[w15, 4:7]  // 11000000-00000110-11100100-00110100
// CHECK-INST: mov     { z20.b - z23.b }, za0v.b[w15, 4:7]
// CHECK-ENCODING: [0x34,0xe4,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006e434 <unknown>

mov     {z28.b - z31.b}, za0v.b[w15, 12:15]  // 11000000-00000110-11100100-01111100
// CHECK-INST: mov     { z28.b - z31.b }, za0v.b[w15, 12:15]
// CHECK-ENCODING: [0x7c,0xe4,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006e47c <unknown>

mov     {z4.b - z7.b}, za0v.b[w12, 4:7]  // 11000000-00000110-10000100-00100100
// CHECK-INST: mov     { z4.b - z7.b }, za0v.b[w12, 4:7]
// CHECK-ENCODING: [0x24,0x84,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0068424 <unknown>

mov     {z0.b - z3.b}, za0v.b[w12, 4:7]  // 11000000-00000110-10000100-00100000
// CHECK-INST: mov     { z0.b - z3.b }, za0v.b[w12, 4:7]
// CHECK-ENCODING: [0x20,0x84,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0068420 <unknown>

mov     {z24.b - z27.b}, za0v.b[w14, 12:15]  // 11000000-00000110-11000100-01111000
// CHECK-INST: mov     { z24.b - z27.b }, za0v.b[w14, 12:15]
// CHECK-ENCODING: [0x78,0xc4,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006c478 <unknown>

mov     {z16.b - z19.b}, za0v.b[w14, 4:7]  // 11000000-00000110-11000100-00110000
// CHECK-INST: mov     { z16.b - z19.b }, za0v.b[w14, 4:7]
// CHECK-ENCODING: [0x30,0xc4,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006c430 <unknown>

mov     {z28.b - z31.b}, za0v.b[w12, 8:11]  // 11000000-00000110-10000100-01011100
// CHECK-INST: mov     { z28.b - z31.b }, za0v.b[w12, 8:11]
// CHECK-ENCODING: [0x5c,0x84,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006845c <unknown>

mov     {z0.b - z3.b}, za0v.b[w15, 4:7]  // 11000000-00000110-11100100-00100000
// CHECK-INST: mov     { z0.b - z3.b }, za0v.b[w15, 4:7]
// CHECK-ENCODING: [0x20,0xe4,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006e420 <unknown>

mov     {z4.b - z7.b}, za0v.b[w13, 0:3]  // 11000000-00000110-10100100-00000100
// CHECK-INST: mov     { z4.b - z7.b }, za0v.b[w13, 0:3]
// CHECK-ENCODING: [0x04,0xa4,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c006a404 <unknown>


mova    za0h.b[w12, 0:3], {z0.b - z3.b}  // 11000000-00000100-00000100-00000000
// CHECK-INST: mov     za0h.b[w12, 0:3], { z0.b - z3.b }
// CHECK-ENCODING: [0x00,0x04,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040400 <unknown>

mova    za0h.b[w14, 4:7], {z8.b - z11.b}  // 11000000-00000100-01000101-00000001
// CHECK-INST: mov     za0h.b[w14, 4:7], { z8.b - z11.b }
// CHECK-ENCODING: [0x01,0x45,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044501 <unknown>

mova    za0h.b[w15, 12:15], {z12.b - z15.b}  // 11000000-00000100-01100101-10000011
// CHECK-INST: mov     za0h.b[w15, 12:15], { z12.b - z15.b }
// CHECK-ENCODING: [0x83,0x65,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046583 <unknown>

mova    za0h.b[w15, 12:15], {z28.b - z31.b}  // 11000000-00000100-01100111-10000011
// CHECK-INST: mov     za0h.b[w15, 12:15], { z28.b - z31.b }
// CHECK-ENCODING: [0x83,0x67,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046783 <unknown>

mova    za0h.b[w12, 4:7], {z16.b - z19.b}  // 11000000-00000100-00000110-00000001
// CHECK-INST: mov     za0h.b[w12, 4:7], { z16.b - z19.b }
// CHECK-ENCODING: [0x01,0x06,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040601 <unknown>

mova    za0h.b[w12, 4:7], {z0.b - z3.b}  // 11000000-00000100-00000100-00000001
// CHECK-INST: mov     za0h.b[w12, 4:7], { z0.b - z3.b }
// CHECK-ENCODING: [0x01,0x04,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040401 <unknown>

mova    za0h.b[w14, 0:3], {z16.b - z19.b}  // 11000000-00000100-01000110-00000000
// CHECK-INST: mov     za0h.b[w14, 0:3], { z16.b - z19.b }
// CHECK-ENCODING: [0x00,0x46,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044600 <unknown>

mova    za0h.b[w12, 0:3], {z12.b - z15.b}  // 11000000-00000100-00000101-10000000
// CHECK-INST: mov     za0h.b[w12, 0:3], { z12.b - z15.b }
// CHECK-ENCODING: [0x80,0x05,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040580 <unknown>

mova    za0h.b[w14, 4:7], {z0.b - z3.b}  // 11000000-00000100-01000100-00000001
// CHECK-INST: mov     za0h.b[w14, 4:7], { z0.b - z3.b }
// CHECK-ENCODING: [0x01,0x44,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044401 <unknown>

mova    za0h.b[w12, 4:7], {z20.b - z23.b}  // 11000000-00000100-00000110-10000001
// CHECK-INST: mov     za0h.b[w12, 4:7], { z20.b - z23.b }
// CHECK-ENCODING: [0x81,0x06,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040681 <unknown>

mova    za0h.b[w15, 8:11], {z8.b - z11.b}  // 11000000-00000100-01100101-00000010
// CHECK-INST: mov     za0h.b[w15, 8:11], { z8.b - z11.b }
// CHECK-ENCODING: [0x02,0x65,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046502 <unknown>

mova    za0h.b[w13, 12:15], {z12.b - z15.b}  // 11000000-00000100-00100101-10000011
// CHECK-INST: mov     za0h.b[w13, 12:15], { z12.b - z15.b }
// CHECK-ENCODING: [0x83,0x25,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0042583 <unknown>

// Aliases

mov     za0h.b[w12, 0:3], {z0.b - z3.b}  // 11000000-00000100-00000100-00000000
// CHECK-INST: mov     za0h.b[w12, 0:3], { z0.b - z3.b }
// CHECK-ENCODING: [0x00,0x04,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040400 <unknown>

mov     za0h.b[w14, 4:7], {z8.b - z11.b}  // 11000000-00000100-01000101-00000001
// CHECK-INST: mov     za0h.b[w14, 4:7], { z8.b - z11.b }
// CHECK-ENCODING: [0x01,0x45,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044501 <unknown>

mov     za0h.b[w15, 12:15], {z12.b - z15.b}  // 11000000-00000100-01100101-10000011
// CHECK-INST: mov     za0h.b[w15, 12:15], { z12.b - z15.b }
// CHECK-ENCODING: [0x83,0x65,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046583 <unknown>

mov     za0h.b[w15, 12:15], {z28.b - z31.b}  // 11000000-00000100-01100111-10000011
// CHECK-INST: mov     za0h.b[w15, 12:15], { z28.b - z31.b }
// CHECK-ENCODING: [0x83,0x67,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046783 <unknown>

mov     za0h.b[w12, 4:7], {z16.b - z19.b}  // 11000000-00000100-00000110-00000001
// CHECK-INST: mov     za0h.b[w12, 4:7], { z16.b - z19.b }
// CHECK-ENCODING: [0x01,0x06,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040601 <unknown>

mov     za0h.b[w12, 4:7], {z0.b - z3.b}  // 11000000-00000100-00000100-00000001
// CHECK-INST: mov     za0h.b[w12, 4:7], { z0.b - z3.b }
// CHECK-ENCODING: [0x01,0x04,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040401 <unknown>

mov     za0h.b[w14, 0:3], {z16.b - z19.b}  // 11000000-00000100-01000110-00000000
// CHECK-INST: mov     za0h.b[w14, 0:3], { z16.b - z19.b }
// CHECK-ENCODING: [0x00,0x46,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044600 <unknown>

mov     za0h.b[w12, 0:3], {z12.b - z15.b}  // 11000000-00000100-00000101-10000000
// CHECK-INST: mov     za0h.b[w12, 0:3], { z12.b - z15.b }
// CHECK-ENCODING: [0x80,0x05,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040580 <unknown>

mov     za0h.b[w14, 4:7], {z0.b - z3.b}  // 11000000-00000100-01000100-00000001
// CHECK-INST: mov     za0h.b[w14, 4:7], { z0.b - z3.b }
// CHECK-ENCODING: [0x01,0x44,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0044401 <unknown>

mov     za0h.b[w12, 4:7], {z20.b - z23.b}  // 11000000-00000100-00000110-10000001
// CHECK-INST: mov     za0h.b[w12, 4:7], { z20.b - z23.b }
// CHECK-ENCODING: [0x81,0x06,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0040681 <unknown>

mov     za0h.b[w15, 8:11], {z8.b - z11.b}  // 11000000-00000100-01100101-00000010
// CHECK-INST: mov     za0h.b[w15, 8:11], { z8.b - z11.b }
// CHECK-ENCODING: [0x02,0x65,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0046502 <unknown>

mov     za0h.b[w13, 12:15], {z12.b - z15.b}  // 11000000-00000100-00100101-10000011
// CHECK-INST: mov     za0h.b[w13, 12:15], { z12.b - z15.b }
// CHECK-ENCODING: [0x83,0x25,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0042583 <unknown>


mova    za0v.b[w12, 0:3], {z0.b - z3.b}  // 11000000-00000100-10000100-00000000
// CHECK-INST: mov     za0v.b[w12, 0:3], { z0.b - z3.b }
// CHECK-ENCODING: [0x00,0x84,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048400 <unknown>

mova    za0v.b[w14, 4:7], {z8.b - z11.b}  // 11000000-00000100-11000101-00000001
// CHECK-INST: mov     za0v.b[w14, 4:7], { z8.b - z11.b }
// CHECK-ENCODING: [0x01,0xc5,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004c501 <unknown>

mova    za0v.b[w15, 12:15], {z12.b - z15.b}  // 11000000-00000100-11100101-10000011
// CHECK-INST: mov     za0v.b[w15, 12:15], { z12.b - z15.b }
// CHECK-ENCODING: [0x83,0xe5,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004e583 <unknown>

mova    za0v.b[w15, 12:15], {z28.b - z31.b}  // 11000000-00000100-11100111-10000011
// CHECK-INST: mov     za0v.b[w15, 12:15], { z28.b - z31.b }
// CHECK-ENCODING: [0x83,0xe7,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004e783 <unknown>

mova    za0v.b[w12, 4:7], {z16.b - z19.b}  // 11000000-00000100-10000110-00000001
// CHECK-INST: mov     za0v.b[w12, 4:7], { z16.b - z19.b }
// CHECK-ENCODING: [0x01,0x86,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048601 <unknown>

mova    za0v.b[w12, 4:7], {z0.b - z3.b}  // 11000000-00000100-10000100-00000001
// CHECK-INST: mov     za0v.b[w12, 4:7], { z0.b - z3.b }
// CHECK-ENCODING: [0x01,0x84,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048401 <unknown>

mova    za0v.b[w14, 0:3], {z16.b - z19.b}  // 11000000-00000100-11000110-00000000
// CHECK-INST: mov     za0v.b[w14, 0:3], { z16.b - z19.b }
// CHECK-ENCODING: [0x00,0xc6,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004c600 <unknown>

mova    za0v.b[w12, 0:3], {z12.b - z15.b}  // 11000000-00000100-10000101-10000000
// CHECK-INST: mov     za0v.b[w12, 0:3], { z12.b - z15.b }
// CHECK-ENCODING: [0x80,0x85,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048580 <unknown>

mova    za0v.b[w14, 4:7], {z0.b - z3.b}  // 11000000-00000100-11000100-00000001
// CHECK-INST: mov     za0v.b[w14, 4:7], { z0.b - z3.b }
// CHECK-ENCODING: [0x01,0xc4,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004c401 <unknown>

mova    za0v.b[w12, 4:7], {z20.b - z23.b}  // 11000000-00000100-10000110-10000001
// CHECK-INST: mov     za0v.b[w12, 4:7], { z20.b - z23.b }
// CHECK-ENCODING: [0x81,0x86,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048681 <unknown>

mova    za0v.b[w15, 8:11], {z8.b - z11.b}  // 11000000-00000100-11100101-00000010
// CHECK-INST: mov     za0v.b[w15, 8:11], { z8.b - z11.b }
// CHECK-ENCODING: [0x02,0xe5,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004e502 <unknown>

mova    za0v.b[w13, 12:15], {z12.b - z15.b}  // 11000000-00000100-10100101-10000011
// CHECK-INST: mov     za0v.b[w13, 12:15], { z12.b - z15.b }
// CHECK-ENCODING: [0x83,0xa5,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004a583 <unknown>

// Aliases

mov     za0v.b[w12, 0:3], {z0.b - z3.b}  // 11000000-00000100-10000100-00000000
// CHECK-INST: mov     za0v.b[w12, 0:3], { z0.b - z3.b }
// CHECK-ENCODING: [0x00,0x84,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048400 <unknown>

mov     za0v.b[w14, 4:7], {z8.b - z11.b}  // 11000000-00000100-11000101-00000001
// CHECK-INST: mov     za0v.b[w14, 4:7], { z8.b - z11.b }
// CHECK-ENCODING: [0x01,0xc5,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004c501 <unknown>

mov     za0v.b[w15, 12:15], {z12.b - z15.b}  // 11000000-00000100-11100101-10000011
// CHECK-INST: mov     za0v.b[w15, 12:15], { z12.b - z15.b }
// CHECK-ENCODING: [0x83,0xe5,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004e583 <unknown>

mov     za0v.b[w15, 12:15], {z28.b - z31.b}  // 11000000-00000100-11100111-10000011
// CHECK-INST: mov     za0v.b[w15, 12:15], { z28.b - z31.b }
// CHECK-ENCODING: [0x83,0xe7,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004e783 <unknown>

mov     za0v.b[w12, 4:7], {z16.b - z19.b}  // 11000000-00000100-10000110-00000001
// CHECK-INST: mov     za0v.b[w12, 4:7], { z16.b - z19.b }
// CHECK-ENCODING: [0x01,0x86,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048601 <unknown>

mov     za0v.b[w12, 4:7], {z0.b - z3.b}  // 11000000-00000100-10000100-00000001
// CHECK-INST: mov     za0v.b[w12, 4:7], { z0.b - z3.b }
// CHECK-ENCODING: [0x01,0x84,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048401 <unknown>

mov     za0v.b[w14, 0:3], {z16.b - z19.b}  // 11000000-00000100-11000110-00000000
// CHECK-INST: mov     za0v.b[w14, 0:3], { z16.b - z19.b }
// CHECK-ENCODING: [0x00,0xc6,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004c600 <unknown>

mov     za0v.b[w12, 0:3], {z12.b - z15.b}  // 11000000-00000100-10000101-10000000
// CHECK-INST: mov     za0v.b[w12, 0:3], { z12.b - z15.b }
// CHECK-ENCODING: [0x80,0x85,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048580 <unknown>

mov     za0v.b[w14, 4:7], {z0.b - z3.b}  // 11000000-00000100-11000100-00000001
// CHECK-INST: mov     za0v.b[w14, 4:7], { z0.b - z3.b }
// CHECK-ENCODING: [0x01,0xc4,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004c401 <unknown>

mov     za0v.b[w12, 4:7], {z20.b - z23.b}  // 11000000-00000100-10000110-10000001
// CHECK-INST: mov     za0v.b[w12, 4:7], { z20.b - z23.b }
// CHECK-ENCODING: [0x81,0x86,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c0048681 <unknown>

mov     za0v.b[w15, 8:11], {z8.b - z11.b}  // 11000000-00000100-11100101-00000010
// CHECK-INST: mov     za0v.b[w15, 8:11], { z8.b - z11.b }
// CHECK-ENCODING: [0x02,0xe5,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004e502 <unknown>

mov     za0v.b[w13, 12:15], {z12.b - z15.b}  // 11000000-00000100-10100101-10000011
// CHECK-INST: mov     za0v.b[w13, 12:15], { z12.b - z15.b }
// CHECK-ENCODING: [0x83,0xa5,0x04,0xc0]
// CHECK-ERROR: instruction requires: sme2
// CHECK-UNKNOWN: c004a583 <unknown>

