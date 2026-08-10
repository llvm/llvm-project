// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=+sme2p1 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2p1 < %s \
// RUN:        | llvm-objdump --no-print-imm-hex -d --mattr=-sme2p1 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p1 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2p1 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

movaz   {z0.d, z1.d}, za.d[w8, 0, vgx2]  // 11000000-00000110-00001010-00000000
// CHECK-INST: movaz   { z0.d, z1.d }, za.d[w8, 0, vgx2]
// CHECK-ENCODING: [0x00,0x0a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060a00 <unknown>

movaz   {z0.d, z1.d}, za.d[w8, 0]  // 11000000-00000110-00001010-00000000
// CHECK-INST: movaz   { z0.d, z1.d }, za.d[w8, 0, vgx2]
// CHECK-ENCODING: [0x00,0x0a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060a00 <unknown>

movaz   {z20.d, z21.d}, za.d[w10, 2, vgx2]  // 11000000-00000110-01001010-01010100
// CHECK-INST: movaz   { z20.d, z21.d }, za.d[w10, 2, vgx2]
// CHECK-ENCODING: [0x54,0x4a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0064a54 <unknown>

movaz   {z20.d, z21.d}, za.d[w10, 2]  // 11000000-00000110-01001010-01010100
// CHECK-INST: movaz   { z20.d, z21.d }, za.d[w10, 2, vgx2]
// CHECK-ENCODING: [0x54,0x4a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0064a54 <unknown>

movaz   {z22.d, z23.d}, za.d[w11, 5, vgx2]  // 11000000-00000110-01101010-10110110
// CHECK-INST: movaz   { z22.d, z23.d }, za.d[w11, 5, vgx2]
// CHECK-ENCODING: [0xb6,0x6a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0066ab6 <unknown>

movaz   {z22.d, z23.d}, za.d[w11, 5]  // 11000000-00000110-01101010-10110110
// CHECK-INST: movaz   { z22.d, z23.d }, za.d[w11, 5, vgx2]
// CHECK-ENCODING: [0xb6,0x6a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0066ab6 <unknown>

movaz   {z30.d, z31.d}, za.d[w11, 7, vgx2]  // 11000000-00000110-01101010-11111110
// CHECK-INST: movaz   { z30.d, z31.d }, za.d[w11, 7, vgx2]
// CHECK-ENCODING: [0xfe,0x6a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0066afe <unknown>

movaz   {z30.d, z31.d}, za.d[w11, 7]  // 11000000-00000110-01101010-11111110
// CHECK-INST: movaz   { z30.d, z31.d }, za.d[w11, 7, vgx2]
// CHECK-ENCODING: [0xfe,0x6a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0066afe <unknown>

movaz   {z4.d, z5.d}, za.d[w8, 1, vgx2]  // 11000000-00000110-00001010-00100100
// CHECK-INST: movaz   { z4.d, z5.d }, za.d[w8, 1, vgx2]
// CHECK-ENCODING: [0x24,0x0a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060a24 <unknown>

movaz   {z4.d, z5.d}, za.d[w8, 1]  // 11000000-00000110-00001010-00100100
// CHECK-INST: movaz   { z4.d, z5.d }, za.d[w8, 1, vgx2]
// CHECK-ENCODING: [0x24,0x0a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060a24 <unknown>

movaz   {z0.d, z1.d}, za.d[w8, 1, vgx2]  // 11000000-00000110-00001010-00100000
// CHECK-INST: movaz   { z0.d, z1.d }, za.d[w8, 1, vgx2]
// CHECK-ENCODING: [0x20,0x0a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060a20 <unknown>

movaz   {z0.d, z1.d}, za.d[w8, 1]  // 11000000-00000110-00001010-00100000
// CHECK-INST: movaz   { z0.d, z1.d }, za.d[w8, 1, vgx2]
// CHECK-ENCODING: [0x20,0x0a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060a20 <unknown>

movaz   {z24.d, z25.d}, za.d[w10, 3, vgx2]  // 11000000-00000110-01001010-01111000
// CHECK-INST: movaz   { z24.d, z25.d }, za.d[w10, 3, vgx2]
// CHECK-ENCODING: [0x78,0x4a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0064a78 <unknown>

movaz   {z24.d, z25.d}, za.d[w10, 3]  // 11000000-00000110-01001010-01111000
// CHECK-INST: movaz   { z24.d, z25.d }, za.d[w10, 3, vgx2]
// CHECK-ENCODING: [0x78,0x4a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0064a78 <unknown>

movaz   {z0.d, z1.d}, za.d[w8, 4, vgx2]  // 11000000-00000110-00001010-10000000
// CHECK-INST: movaz   { z0.d, z1.d }, za.d[w8, 4, vgx2]
// CHECK-ENCODING: [0x80,0x0a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060a80 <unknown>

movaz   {z0.d, z1.d}, za.d[w8, 4]  // 11000000-00000110-00001010-10000000
// CHECK-INST: movaz   { z0.d, z1.d }, za.d[w8, 4, vgx2]
// CHECK-ENCODING: [0x80,0x0a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060a80 <unknown>

movaz   {z16.d, z17.d}, za.d[w10, 1, vgx2]  // 11000000-00000110-01001010-00110000
// CHECK-INST: movaz   { z16.d, z17.d }, za.d[w10, 1, vgx2]
// CHECK-ENCODING: [0x30,0x4a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0064a30 <unknown>

movaz   {z16.d, z17.d}, za.d[w10, 1]  // 11000000-00000110-01001010-00110000
// CHECK-INST: movaz   { z16.d, z17.d }, za.d[w10, 1, vgx2]
// CHECK-ENCODING: [0x30,0x4a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0064a30 <unknown>

movaz   {z28.d, z29.d}, za.d[w8, 6, vgx2]  // 11000000-00000110-00001010-11011100
// CHECK-INST: movaz   { z28.d, z29.d }, za.d[w8, 6, vgx2]
// CHECK-ENCODING: [0xdc,0x0a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060adc <unknown>

movaz   {z28.d, z29.d}, za.d[w8, 6]  // 11000000-00000110-00001010-11011100
// CHECK-INST: movaz   { z28.d, z29.d }, za.d[w8, 6, vgx2]
// CHECK-ENCODING: [0xdc,0x0a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060adc <unknown>

movaz   {z2.d, z3.d}, za.d[w11, 1, vgx2]  // 11000000-00000110-01101010-00100010
// CHECK-INST: movaz   { z2.d, z3.d }, za.d[w11, 1, vgx2]
// CHECK-ENCODING: [0x22,0x6a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0066a22 <unknown>

movaz   {z2.d, z3.d}, za.d[w11, 1]  // 11000000-00000110-01101010-00100010
// CHECK-INST: movaz   { z2.d, z3.d }, za.d[w11, 1, vgx2]
// CHECK-ENCODING: [0x22,0x6a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0066a22 <unknown>

movaz   {z6.d, z7.d}, za.d[w9, 4, vgx2]  // 11000000-00000110-00101010-10000110
// CHECK-INST: movaz   { z6.d, z7.d }, za.d[w9, 4, vgx2]
// CHECK-ENCODING: [0x86,0x2a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0062a86 <unknown>

movaz   {z6.d, z7.d}, za.d[w9, 4]  // 11000000-00000110-00101010-10000110
// CHECK-INST: movaz   { z6.d, z7.d }, za.d[w9, 4, vgx2]
// CHECK-ENCODING: [0x86,0x2a,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0062a86 <unknown>


movaz   {z0.d - z3.d}, za.d[w8, 0, vgx4]  // 11000000-00000110-00001110-00000000
// CHECK-INST: movaz   { z0.d - z3.d }, za.d[w8, 0, vgx4]
// CHECK-ENCODING: [0x00,0x0e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060e00 <unknown>

movaz   {z0.d - z3.d}, za.d[w8, 0]  // 11000000-00000110-00001110-00000000
// CHECK-INST: movaz   { z0.d - z3.d }, za.d[w8, 0, vgx4]
// CHECK-ENCODING: [0x00,0x0e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060e00 <unknown>

movaz   {z20.d - z23.d}, za.d[w10, 2, vgx4]  // 11000000-00000110-01001110-01010100
// CHECK-INST: movaz   { z20.d - z23.d }, za.d[w10, 2, vgx4]
// CHECK-ENCODING: [0x54,0x4e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0064e54 <unknown>

movaz   {z20.d - z23.d}, za.d[w10, 2]  // 11000000-00000110-01001110-01010100
// CHECK-INST: movaz   { z20.d - z23.d }, za.d[w10, 2, vgx4]
// CHECK-ENCODING: [0x54,0x4e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0064e54 <unknown>

movaz   {z20.d - z23.d}, za.d[w11, 5, vgx4]  // 11000000-00000110-01101110-10110100
// CHECK-INST: movaz   { z20.d - z23.d }, za.d[w11, 5, vgx4]
// CHECK-ENCODING: [0xb4,0x6e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0066eb4 <unknown>

movaz   {z20.d - z23.d}, za.d[w11, 5]  // 11000000-00000110-01101110-10110100
// CHECK-INST: movaz   { z20.d - z23.d }, za.d[w11, 5, vgx4]
// CHECK-ENCODING: [0xb4,0x6e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0066eb4 <unknown>

movaz   {z28.d - z31.d}, za.d[w11, 7, vgx4]  // 11000000-00000110-01101110-11111100
// CHECK-INST: movaz   { z28.d - z31.d }, za.d[w11, 7, vgx4]
// CHECK-ENCODING: [0xfc,0x6e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0066efc <unknown>

movaz   {z28.d - z31.d}, za.d[w11, 7]  // 11000000-00000110-01101110-11111100
// CHECK-INST: movaz   { z28.d - z31.d }, za.d[w11, 7, vgx4]
// CHECK-ENCODING: [0xfc,0x6e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0066efc <unknown>

movaz   {z4.d - z7.d}, za.d[w8, 1, vgx4]  // 11000000-00000110-00001110-00100100
// CHECK-INST: movaz   { z4.d - z7.d }, za.d[w8, 1, vgx4]
// CHECK-ENCODING: [0x24,0x0e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060e24 <unknown>

movaz   {z4.d - z7.d}, za.d[w8, 1]  // 11000000-00000110-00001110-00100100
// CHECK-INST: movaz   { z4.d - z7.d }, za.d[w8, 1, vgx4]
// CHECK-ENCODING: [0x24,0x0e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060e24 <unknown>

movaz   {z0.d - z3.d}, za.d[w8, 1, vgx4]  // 11000000-00000110-00001110-00100000
// CHECK-INST: movaz   { z0.d - z3.d }, za.d[w8, 1, vgx4]
// CHECK-ENCODING: [0x20,0x0e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060e20 <unknown>

movaz   {z0.d - z3.d}, za.d[w8, 1]  // 11000000-00000110-00001110-00100000
// CHECK-INST: movaz   { z0.d - z3.d }, za.d[w8, 1, vgx4]
// CHECK-ENCODING: [0x20,0x0e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060e20 <unknown>

movaz   {z24.d - z27.d}, za.d[w10, 3, vgx4]  // 11000000-00000110-01001110-01111000
// CHECK-INST: movaz   { z24.d - z27.d }, za.d[w10, 3, vgx4]
// CHECK-ENCODING: [0x78,0x4e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0064e78 <unknown>

movaz   {z24.d - z27.d}, za.d[w10, 3]  // 11000000-00000110-01001110-01111000
// CHECK-INST: movaz   { z24.d - z27.d }, za.d[w10, 3, vgx4]
// CHECK-ENCODING: [0x78,0x4e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0064e78 <unknown>

movaz   {z0.d - z3.d}, za.d[w8, 4, vgx4]  // 11000000-00000110-00001110-10000000
// CHECK-INST: movaz   { z0.d - z3.d }, za.d[w8, 4, vgx4]
// CHECK-ENCODING: [0x80,0x0e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060e80 <unknown>

movaz   {z0.d - z3.d}, za.d[w8, 4]  // 11000000-00000110-00001110-10000000
// CHECK-INST: movaz   { z0.d - z3.d }, za.d[w8, 4, vgx4]
// CHECK-ENCODING: [0x80,0x0e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060e80 <unknown>

movaz   {z16.d - z19.d}, za.d[w10, 1, vgx4]  // 11000000-00000110-01001110-00110000
// CHECK-INST: movaz   { z16.d - z19.d }, za.d[w10, 1, vgx4]
// CHECK-ENCODING: [0x30,0x4e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0064e30 <unknown>

movaz   {z16.d - z19.d}, za.d[w10, 1]  // 11000000-00000110-01001110-00110000
// CHECK-INST: movaz   { z16.d - z19.d }, za.d[w10, 1, vgx4]
// CHECK-ENCODING: [0x30,0x4e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0064e30 <unknown>

movaz   {z28.d - z31.d}, za.d[w8, 6, vgx4]  // 11000000-00000110-00001110-11011100
// CHECK-INST: movaz   { z28.d - z31.d }, za.d[w8, 6, vgx4]
// CHECK-ENCODING: [0xdc,0x0e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060edc <unknown>

movaz   {z28.d - z31.d}, za.d[w8, 6]  // 11000000-00000110-00001110-11011100
// CHECK-INST: movaz   { z28.d - z31.d }, za.d[w8, 6, vgx4]
// CHECK-ENCODING: [0xdc,0x0e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0060edc <unknown>

movaz   {z0.d - z3.d}, za.d[w11, 1, vgx4]  // 11000000-00000110-01101110-00100000
// CHECK-INST: movaz   { z0.d - z3.d }, za.d[w11, 1, vgx4]
// CHECK-ENCODING: [0x20,0x6e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0066e20 <unknown>

movaz   {z0.d - z3.d}, za.d[w11, 1]  // 11000000-00000110-01101110-00100000
// CHECK-INST: movaz   { z0.d - z3.d }, za.d[w11, 1, vgx4]
// CHECK-ENCODING: [0x20,0x6e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0066e20 <unknown>

movaz   {z4.d - z7.d}, za.d[w9, 4, vgx4]  // 11000000-00000110-00101110-10000100
// CHECK-INST: movaz   { z4.d - z7.d }, za.d[w9, 4, vgx4]
// CHECK-ENCODING: [0x84,0x2e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0062e84 <unknown>

movaz   {z4.d - z7.d}, za.d[w9, 4]  // 11000000-00000110-00101110-10000100
// CHECK-INST: movaz   { z4.d - z7.d }, za.d[w9, 4, vgx4]
// CHECK-ENCODING: [0x84,0x2e,0x06,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0062e84 <unknown>


movaz   z0.q, za0h.q[w12, 0]  // 11000000-11000011-00000010-00000000
// CHECK-INST: movaz   z0.q, za0h.q[w12, 0]
// CHECK-ENCODING: [0x00,0x02,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c30200 <unknown>

movaz   z21.q, za10h.q[w14, 0]  // 11000000-11000011-01000011-01010101
// CHECK-INST: movaz   z21.q, za10h.q[w14, 0]
// CHECK-ENCODING: [0x55,0x43,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c34355 <unknown>

movaz   z23.q, za13h.q[w15, 0]  // 11000000-11000011-01100011-10110111
// CHECK-INST: movaz   z23.q, za13h.q[w15, 0]
// CHECK-ENCODING: [0xb7,0x63,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c363b7 <unknown>

movaz   z31.q, za15h.q[w15, 0]  // 11000000-11000011-01100011-11111111
// CHECK-INST: movaz   z31.q, za15h.q[w15, 0]
// CHECK-ENCODING: [0xff,0x63,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c363ff <unknown>

movaz   z5.q, za1h.q[w12, 0]  // 11000000-11000011-00000010-00100101
// CHECK-INST: movaz   z5.q, za1h.q[w12, 0]
// CHECK-ENCODING: [0x25,0x02,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c30225 <unknown>

movaz   z1.q, za1h.q[w12, 0]  // 11000000-11000011-00000010-00100001
// CHECK-INST: movaz   z1.q, za1h.q[w12, 0]
// CHECK-ENCODING: [0x21,0x02,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c30221 <unknown>

movaz   z24.q, za3h.q[w14, 0]  // 11000000-11000011-01000010-01111000
// CHECK-INST: movaz   z24.q, za3h.q[w14, 0]
// CHECK-ENCODING: [0x78,0x42,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c34278 <unknown>

movaz   z0.q, za12h.q[w12, 0]  // 11000000-11000011-00000011-10000000
// CHECK-INST: movaz   z0.q, za12h.q[w12, 0]
// CHECK-ENCODING: [0x80,0x03,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c30380 <unknown>

movaz   z17.q, za1h.q[w14, 0]  // 11000000-11000011-01000010-00110001
// CHECK-INST: movaz   z17.q, za1h.q[w14, 0]
// CHECK-ENCODING: [0x31,0x42,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c34231 <unknown>

movaz   z29.q, za6h.q[w12, 0]  // 11000000-11000011-00000010-11011101
// CHECK-INST: movaz   z29.q, za6h.q[w12, 0]
// CHECK-ENCODING: [0xdd,0x02,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c302dd <unknown>

movaz   z2.q, za9h.q[w15, 0]  // 11000000-11000011-01100011-00100010
// CHECK-INST: movaz   z2.q, za9h.q[w15, 0]
// CHECK-ENCODING: [0x22,0x63,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c36322 <unknown>

movaz   z7.q, za12h.q[w13, 0]  // 11000000-11000011-00100011-10000111
// CHECK-INST: movaz   z7.q, za12h.q[w13, 0]
// CHECK-ENCODING: [0x87,0x23,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c32387 <unknown>

movaz   z0.q, za0v.q[w12, 0]  // 11000000-11000011-10000010-00000000
// CHECK-INST: movaz   z0.q, za0v.q[w12, 0]
// CHECK-ENCODING: [0x00,0x82,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c38200 <unknown>

movaz   z21.q, za10v.q[w14, 0]  // 11000000-11000011-11000011-01010101
// CHECK-INST: movaz   z21.q, za10v.q[w14, 0]
// CHECK-ENCODING: [0x55,0xc3,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c3c355 <unknown>

movaz   z23.q, za13v.q[w15, 0]  // 11000000-11000011-11100011-10110111
// CHECK-INST: movaz   z23.q, za13v.q[w15, 0]
// CHECK-ENCODING: [0xb7,0xe3,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c3e3b7 <unknown>

movaz   z31.q, za15v.q[w15, 0]  // 11000000-11000011-11100011-11111111
// CHECK-INST: movaz   z31.q, za15v.q[w15, 0]
// CHECK-ENCODING: [0xff,0xe3,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c3e3ff <unknown>

movaz   z5.q, za1v.q[w12, 0]  // 11000000-11000011-10000010-00100101
// CHECK-INST: movaz   z5.q, za1v.q[w12, 0]
// CHECK-ENCODING: [0x25,0x82,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c38225 <unknown>

movaz   z1.q, za1v.q[w12, 0]  // 11000000-11000011-10000010-00100001
// CHECK-INST: movaz   z1.q, za1v.q[w12, 0]
// CHECK-ENCODING: [0x21,0x82,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c38221 <unknown>

movaz   z24.q, za3v.q[w14, 0]  // 11000000-11000011-11000010-01111000
// CHECK-INST: movaz   z24.q, za3v.q[w14, 0]
// CHECK-ENCODING: [0x78,0xc2,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c3c278 <unknown>

movaz   z0.q, za12v.q[w12, 0]  // 11000000-11000011-10000011-10000000
// CHECK-INST: movaz   z0.q, za12v.q[w12, 0]
// CHECK-ENCODING: [0x80,0x83,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c38380 <unknown>

movaz   z17.q, za1v.q[w14, 0]  // 11000000-11000011-11000010-00110001
// CHECK-INST: movaz   z17.q, za1v.q[w14, 0]
// CHECK-ENCODING: [0x31,0xc2,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c3c231 <unknown>

movaz   z29.q, za6v.q[w12, 0]  // 11000000-11000011-10000010-11011101
// CHECK-INST: movaz   z29.q, za6v.q[w12, 0]
// CHECK-ENCODING: [0xdd,0x82,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c382dd <unknown>

movaz   z2.q, za9v.q[w15, 0]  // 11000000-11000011-11100011-00100010
// CHECK-INST: movaz   z2.q, za9v.q[w15, 0]
// CHECK-ENCODING: [0x22,0xe3,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c3e322 <unknown>

movaz   z7.q, za12v.q[w13, 0]  // 11000000-11000011-10100011-10000111
// CHECK-INST: movaz   z7.q, za12v.q[w13, 0]
// CHECK-ENCODING: [0x87,0xa3,0xc3,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c3a387 <unknown>

movaz   z0.h, za0h.h[w12, 0]  // 11000000-01000010-00000010-00000000
// CHECK-INST: movaz   z0.h, za0h.h[w12, 0]
// CHECK-ENCODING: [0x00,0x02,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0420200 <unknown>

movaz   z21.h, za1h.h[w14, 2]  // 11000000-01000010-01000011-01010101
// CHECK-INST: movaz   z21.h, za1h.h[w14, 2]
// CHECK-ENCODING: [0x55,0x43,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0424355 <unknown>

movaz   z23.h, za1h.h[w15, 5]  // 11000000-01000010-01100011-10110111
// CHECK-INST: movaz   z23.h, za1h.h[w15, 5]
// CHECK-ENCODING: [0xb7,0x63,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c04263b7 <unknown>

movaz   z31.h, za1h.h[w15, 7]  // 11000000-01000010-01100011-11111111
// CHECK-INST: movaz   z31.h, za1h.h[w15, 7]
// CHECK-ENCODING: [0xff,0x63,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c04263ff <unknown>

movaz   z5.h, za0h.h[w12, 1]  // 11000000-01000010-00000010-00100101
// CHECK-INST: movaz   z5.h, za0h.h[w12, 1]
// CHECK-ENCODING: [0x25,0x02,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0420225 <unknown>

movaz   z1.h, za0h.h[w12, 1]  // 11000000-01000010-00000010-00100001
// CHECK-INST: movaz   z1.h, za0h.h[w12, 1]
// CHECK-ENCODING: [0x21,0x02,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0420221 <unknown>

movaz   z24.h, za0h.h[w14, 3]  // 11000000-01000010-01000010-01111000
// CHECK-INST: movaz   z24.h, za0h.h[w14, 3]
// CHECK-ENCODING: [0x78,0x42,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0424278 <unknown>

movaz   z0.h, za1h.h[w12, 4]  // 11000000-01000010-00000011-10000000
// CHECK-INST: movaz   z0.h, za1h.h[w12, 4]
// CHECK-ENCODING: [0x80,0x03,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0420380 <unknown>

movaz   z17.h, za0h.h[w14, 1]  // 11000000-01000010-01000010-00110001
// CHECK-INST: movaz   z17.h, za0h.h[w14, 1]
// CHECK-ENCODING: [0x31,0x42,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0424231 <unknown>

movaz   z29.h, za0h.h[w12, 6]  // 11000000-01000010-00000010-11011101
// CHECK-INST: movaz   z29.h, za0h.h[w12, 6]
// CHECK-ENCODING: [0xdd,0x02,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c04202dd <unknown>

movaz   z2.h, za1h.h[w15, 1]  // 11000000-01000010-01100011-00100010
// CHECK-INST: movaz   z2.h, za1h.h[w15, 1]
// CHECK-ENCODING: [0x22,0x63,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0426322 <unknown>

movaz   z7.h, za1h.h[w13, 4]  // 11000000-01000010-00100011-10000111
// CHECK-INST: movaz   z7.h, za1h.h[w13, 4]
// CHECK-ENCODING: [0x87,0x23,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0422387 <unknown>

movaz   z0.h, za0v.h[w12, 0]  // 11000000-01000010-10000010-00000000
// CHECK-INST: movaz   z0.h, za0v.h[w12, 0]
// CHECK-ENCODING: [0x00,0x82,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0428200 <unknown>

movaz   z21.h, za1v.h[w14, 2]  // 11000000-01000010-11000011-01010101
// CHECK-INST: movaz   z21.h, za1v.h[w14, 2]
// CHECK-ENCODING: [0x55,0xc3,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c042c355 <unknown>

movaz   z23.h, za1v.h[w15, 5]  // 11000000-01000010-11100011-10110111
// CHECK-INST: movaz   z23.h, za1v.h[w15, 5]
// CHECK-ENCODING: [0xb7,0xe3,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c042e3b7 <unknown>

movaz   z31.h, za1v.h[w15, 7]  // 11000000-01000010-11100011-11111111
// CHECK-INST: movaz   z31.h, za1v.h[w15, 7]
// CHECK-ENCODING: [0xff,0xe3,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c042e3ff <unknown>

movaz   z5.h, za0v.h[w12, 1]  // 11000000-01000010-10000010-00100101
// CHECK-INST: movaz   z5.h, za0v.h[w12, 1]
// CHECK-ENCODING: [0x25,0x82,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0428225 <unknown>

movaz   z1.h, za0v.h[w12, 1]  // 11000000-01000010-10000010-00100001
// CHECK-INST: movaz   z1.h, za0v.h[w12, 1]
// CHECK-ENCODING: [0x21,0x82,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0428221 <unknown>

movaz   z24.h, za0v.h[w14, 3]  // 11000000-01000010-11000010-01111000
// CHECK-INST: movaz   z24.h, za0v.h[w14, 3]
// CHECK-ENCODING: [0x78,0xc2,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c042c278 <unknown>

movaz   z0.h, za1v.h[w12, 4]  // 11000000-01000010-10000011-10000000
// CHECK-INST: movaz   z0.h, za1v.h[w12, 4]
// CHECK-ENCODING: [0x80,0x83,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0428380 <unknown>

movaz   z17.h, za0v.h[w14, 1]  // 11000000-01000010-11000010-00110001
// CHECK-INST: movaz   z17.h, za0v.h[w14, 1]
// CHECK-ENCODING: [0x31,0xc2,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c042c231 <unknown>

movaz   z29.h, za0v.h[w12, 6]  // 11000000-01000010-10000010-11011101
// CHECK-INST: movaz   z29.h, za0v.h[w12, 6]
// CHECK-ENCODING: [0xdd,0x82,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c04282dd <unknown>

movaz   z2.h, za1v.h[w15, 1]  // 11000000-01000010-11100011-00100010
// CHECK-INST: movaz   z2.h, za1v.h[w15, 1]
// CHECK-ENCODING: [0x22,0xe3,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c042e322 <unknown>

movaz   z7.h, za1v.h[w13, 4]  // 11000000-01000010-10100011-10000111
// CHECK-INST: movaz   z7.h, za1v.h[w13, 4]
// CHECK-ENCODING: [0x87,0xa3,0x42,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c042a387 <unknown>

movaz   z0.s, za0h.s[w12, 0]  // 11000000-10000010-00000010-00000000
// CHECK-INST: movaz   z0.s, za0h.s[w12, 0]
// CHECK-ENCODING: [0x00,0x02,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0820200 <unknown>

movaz   z21.s, za2h.s[w14, 2]  // 11000000-10000010-01000011-01010101
// CHECK-INST: movaz   z21.s, za2h.s[w14, 2]
// CHECK-ENCODING: [0x55,0x43,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0824355 <unknown>

movaz   z23.s, za3h.s[w15, 1]  // 11000000-10000010-01100011-10110111
// CHECK-INST: movaz   z23.s, za3h.s[w15, 1]
// CHECK-ENCODING: [0xb7,0x63,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c08263b7 <unknown>

movaz   z31.s, za3h.s[w15, 3]  // 11000000-10000010-01100011-11111111
// CHECK-INST: movaz   z31.s, za3h.s[w15, 3]
// CHECK-ENCODING: [0xff,0x63,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c08263ff <unknown>

movaz   z5.s, za0h.s[w12, 1]  // 11000000-10000010-00000010-00100101
// CHECK-INST: movaz   z5.s, za0h.s[w12, 1]
// CHECK-ENCODING: [0x25,0x02,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0820225 <unknown>

movaz   z1.s, za0h.s[w12, 1]  // 11000000-10000010-00000010-00100001
// CHECK-INST: movaz   z1.s, za0h.s[w12, 1]
// CHECK-ENCODING: [0x21,0x02,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0820221 <unknown>

movaz   z24.s, za0h.s[w14, 3]  // 11000000-10000010-01000010-01111000
// CHECK-INST: movaz   z24.s, za0h.s[w14, 3]
// CHECK-ENCODING: [0x78,0x42,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0824278 <unknown>

movaz   z0.s, za3h.s[w12, 0]  // 11000000-10000010-00000011-10000000
// CHECK-INST: movaz   z0.s, za3h.s[w12, 0]
// CHECK-ENCODING: [0x80,0x03,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0820380 <unknown>

movaz   z17.s, za0h.s[w14, 1]  // 11000000-10000010-01000010-00110001
// CHECK-INST: movaz   z17.s, za0h.s[w14, 1]
// CHECK-ENCODING: [0x31,0x42,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0824231 <unknown>

movaz   z29.s, za1h.s[w12, 2]  // 11000000-10000010-00000010-11011101
// CHECK-INST: movaz   z29.s, za1h.s[w12, 2]
// CHECK-ENCODING: [0xdd,0x02,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c08202dd <unknown>

movaz   z2.s, za2h.s[w15, 1]  // 11000000-10000010-01100011-00100010
// CHECK-INST: movaz   z2.s, za2h.s[w15, 1]
// CHECK-ENCODING: [0x22,0x63,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0826322 <unknown>

movaz   z7.s, za3h.s[w13, 0]  // 11000000-10000010-00100011-10000111
// CHECK-INST: movaz   z7.s, za3h.s[w13, 0]
// CHECK-ENCODING: [0x87,0x23,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0822387 <unknown>

movaz   z0.s, za0v.s[w12, 0]  // 11000000-10000010-10000010-00000000
// CHECK-INST: movaz   z0.s, za0v.s[w12, 0]
// CHECK-ENCODING: [0x00,0x82,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0828200 <unknown>

movaz   z21.s, za2v.s[w14, 2]  // 11000000-10000010-11000011-01010101
// CHECK-INST: movaz   z21.s, za2v.s[w14, 2]
// CHECK-ENCODING: [0x55,0xc3,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c082c355 <unknown>

movaz   z23.s, za3v.s[w15, 1]  // 11000000-10000010-11100011-10110111
// CHECK-INST: movaz   z23.s, za3v.s[w15, 1]
// CHECK-ENCODING: [0xb7,0xe3,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c082e3b7 <unknown>

movaz   z31.s, za3v.s[w15, 3]  // 11000000-10000010-11100011-11111111
// CHECK-INST: movaz   z31.s, za3v.s[w15, 3]
// CHECK-ENCODING: [0xff,0xe3,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c082e3ff <unknown>

movaz   z5.s, za0v.s[w12, 1]  // 11000000-10000010-10000010-00100101
// CHECK-INST: movaz   z5.s, za0v.s[w12, 1]
// CHECK-ENCODING: [0x25,0x82,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0828225 <unknown>

movaz   z1.s, za0v.s[w12, 1]  // 11000000-10000010-10000010-00100001
// CHECK-INST: movaz   z1.s, za0v.s[w12, 1]
// CHECK-ENCODING: [0x21,0x82,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0828221 <unknown>

movaz   z24.s, za0v.s[w14, 3]  // 11000000-10000010-11000010-01111000
// CHECK-INST: movaz   z24.s, za0v.s[w14, 3]
// CHECK-ENCODING: [0x78,0xc2,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c082c278 <unknown>

movaz   z0.s, za3v.s[w12, 0]  // 11000000-10000010-10000011-10000000
// CHECK-INST: movaz   z0.s, za3v.s[w12, 0]
// CHECK-ENCODING: [0x80,0x83,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0828380 <unknown>

movaz   z17.s, za0v.s[w14, 1]  // 11000000-10000010-11000010-00110001
// CHECK-INST: movaz   z17.s, za0v.s[w14, 1]
// CHECK-ENCODING: [0x31,0xc2,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c082c231 <unknown>

movaz   z29.s, za1v.s[w12, 2]  // 11000000-10000010-10000010-11011101
// CHECK-INST: movaz   z29.s, za1v.s[w12, 2]
// CHECK-ENCODING: [0xdd,0x82,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c08282dd <unknown>

movaz   z2.s, za2v.s[w15, 1]  // 11000000-10000010-11100011-00100010
// CHECK-INST: movaz   z2.s, za2v.s[w15, 1]
// CHECK-ENCODING: [0x22,0xe3,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c082e322 <unknown>

movaz   z7.s, za3v.s[w13, 0]  // 11000000-10000010-10100011-10000111
// CHECK-INST: movaz   z7.s, za3v.s[w13, 0]
// CHECK-ENCODING: [0x87,0xa3,0x82,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c082a387 <unknown>

movaz   z0.d, za0h.d[w12, 0]  // 11000000-11000010-00000010-00000000
// CHECK-INST: movaz   z0.d, za0h.d[w12, 0]
// CHECK-ENCODING: [0x00,0x02,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c20200 <unknown>

movaz   z21.d, za5h.d[w14, 0]  // 11000000-11000010-01000011-01010101
// CHECK-INST: movaz   z21.d, za5h.d[w14, 0]
// CHECK-ENCODING: [0x55,0x43,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c24355 <unknown>

movaz   z23.d, za6h.d[w15, 1]  // 11000000-11000010-01100011-10110111
// CHECK-INST: movaz   z23.d, za6h.d[w15, 1]
// CHECK-ENCODING: [0xb7,0x63,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c263b7 <unknown>

movaz   z31.d, za7h.d[w15, 1]  // 11000000-11000010-01100011-11111111
// CHECK-INST: movaz   z31.d, za7h.d[w15, 1]
// CHECK-ENCODING: [0xff,0x63,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c263ff <unknown>

movaz   z5.d, za0h.d[w12, 1]  // 11000000-11000010-00000010-00100101
// CHECK-INST: movaz   z5.d, za0h.d[w12, 1]
// CHECK-ENCODING: [0x25,0x02,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c20225 <unknown>

movaz   z1.d, za0h.d[w12, 1]  // 11000000-11000010-00000010-00100001
// CHECK-INST: movaz   z1.d, za0h.d[w12, 1]
// CHECK-ENCODING: [0x21,0x02,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c20221 <unknown>

movaz   z24.d, za1h.d[w14, 1]  // 11000000-11000010-01000010-01111000
// CHECK-INST: movaz   z24.d, za1h.d[w14, 1]
// CHECK-ENCODING: [0x78,0x42,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c24278 <unknown>

movaz   z0.d, za6h.d[w12, 0]  // 11000000-11000010-00000011-10000000
// CHECK-INST: movaz   z0.d, za6h.d[w12, 0]
// CHECK-ENCODING: [0x80,0x03,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c20380 <unknown>

movaz   z17.d, za0h.d[w14, 1]  // 11000000-11000010-01000010-00110001
// CHECK-INST: movaz   z17.d, za0h.d[w14, 1]
// CHECK-ENCODING: [0x31,0x42,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c24231 <unknown>

movaz   z29.d, za3h.d[w12, 0]  // 11000000-11000010-00000010-11011101
// CHECK-INST: movaz   z29.d, za3h.d[w12, 0]
// CHECK-ENCODING: [0xdd,0x02,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c202dd <unknown>

movaz   z2.d, za4h.d[w15, 1]  // 11000000-11000010-01100011-00100010
// CHECK-INST: movaz   z2.d, za4h.d[w15, 1]
// CHECK-ENCODING: [0x22,0x63,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c26322 <unknown>

movaz   z7.d, za6h.d[w13, 0]  // 11000000-11000010-00100011-10000111
// CHECK-INST: movaz   z7.d, za6h.d[w13, 0]
// CHECK-ENCODING: [0x87,0x23,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c22387 <unknown>

movaz   z0.d, za0v.d[w12, 0]  // 11000000-11000010-10000010-00000000
// CHECK-INST: movaz   z0.d, za0v.d[w12, 0]
// CHECK-ENCODING: [0x00,0x82,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c28200 <unknown>

movaz   z21.d, za5v.d[w14, 0]  // 11000000-11000010-11000011-01010101
// CHECK-INST: movaz   z21.d, za5v.d[w14, 0]
// CHECK-ENCODING: [0x55,0xc3,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c2c355 <unknown>

movaz   z23.d, za6v.d[w15, 1]  // 11000000-11000010-11100011-10110111
// CHECK-INST: movaz   z23.d, za6v.d[w15, 1]
// CHECK-ENCODING: [0xb7,0xe3,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c2e3b7 <unknown>

movaz   z31.d, za7v.d[w15, 1]  // 11000000-11000010-11100011-11111111
// CHECK-INST: movaz   z31.d, za7v.d[w15, 1]
// CHECK-ENCODING: [0xff,0xe3,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c2e3ff <unknown>

movaz   z5.d, za0v.d[w12, 1]  // 11000000-11000010-10000010-00100101
// CHECK-INST: movaz   z5.d, za0v.d[w12, 1]
// CHECK-ENCODING: [0x25,0x82,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c28225 <unknown>

movaz   z1.d, za0v.d[w12, 1]  // 11000000-11000010-10000010-00100001
// CHECK-INST: movaz   z1.d, za0v.d[w12, 1]
// CHECK-ENCODING: [0x21,0x82,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c28221 <unknown>

movaz   z24.d, za1v.d[w14, 1]  // 11000000-11000010-11000010-01111000
// CHECK-INST: movaz   z24.d, za1v.d[w14, 1]
// CHECK-ENCODING: [0x78,0xc2,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c2c278 <unknown>

movaz   z0.d, za6v.d[w12, 0]  // 11000000-11000010-10000011-10000000
// CHECK-INST: movaz   z0.d, za6v.d[w12, 0]
// CHECK-ENCODING: [0x80,0x83,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c28380 <unknown>

movaz   z17.d, za0v.d[w14, 1]  // 11000000-11000010-11000010-00110001
// CHECK-INST: movaz   z17.d, za0v.d[w14, 1]
// CHECK-ENCODING: [0x31,0xc2,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c2c231 <unknown>

movaz   z29.d, za3v.d[w12, 0]  // 11000000-11000010-10000010-11011101
// CHECK-INST: movaz   z29.d, za3v.d[w12, 0]
// CHECK-ENCODING: [0xdd,0x82,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c282dd <unknown>

movaz   z2.d, za4v.d[w15, 1]  // 11000000-11000010-11100011-00100010
// CHECK-INST: movaz   z2.d, za4v.d[w15, 1]
// CHECK-ENCODING: [0x22,0xe3,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c2e322 <unknown>

movaz   z7.d, za6v.d[w13, 0]  // 11000000-11000010-10100011-10000111
// CHECK-INST: movaz   z7.d, za6v.d[w13, 0]
// CHECK-ENCODING: [0x87,0xa3,0xc2,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0c2a387 <unknown>

movaz   z0.b, za0h.b[w12, 0]  // 11000000-00000010-00000010-00000000
// CHECK-INST: movaz   z0.b, za0h.b[w12, 0]
// CHECK-ENCODING: [0x00,0x02,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0020200 <unknown>

movaz   z21.b, za0h.b[w14, 10]  // 11000000-00000010-01000011-01010101
// CHECK-INST: movaz   z21.b, za0h.b[w14, 10]
// CHECK-ENCODING: [0x55,0x43,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0024355 <unknown>

movaz   z23.b, za0h.b[w15, 13]  // 11000000-00000010-01100011-10110111
// CHECK-INST: movaz   z23.b, za0h.b[w15, 13]
// CHECK-ENCODING: [0xb7,0x63,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00263b7 <unknown>

movaz   z31.b, za0h.b[w15, 15]  // 11000000-00000010-01100011-11111111
// CHECK-INST: movaz   z31.b, za0h.b[w15, 15]
// CHECK-ENCODING: [0xff,0x63,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00263ff <unknown>

movaz   z5.b, za0h.b[w12, 1]  // 11000000-00000010-00000010-00100101
// CHECK-INST: movaz   z5.b, za0h.b[w12, 1]
// CHECK-ENCODING: [0x25,0x02,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0020225 <unknown>

movaz   z1.b, za0h.b[w12, 1]  // 11000000-00000010-00000010-00100001
// CHECK-INST: movaz   z1.b, za0h.b[w12, 1]
// CHECK-ENCODING: [0x21,0x02,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0020221 <unknown>

movaz   z24.b, za0h.b[w14, 3]  // 11000000-00000010-01000010-01111000
// CHECK-INST: movaz   z24.b, za0h.b[w14, 3]
// CHECK-ENCODING: [0x78,0x42,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0024278 <unknown>

movaz   z0.b, za0h.b[w12, 12]  // 11000000-00000010-00000011-10000000
// CHECK-INST: movaz   z0.b, za0h.b[w12, 12]
// CHECK-ENCODING: [0x80,0x03,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0020380 <unknown>

movaz   z17.b, za0h.b[w14, 1]  // 11000000-00000010-01000010-00110001
// CHECK-INST: movaz   z17.b, za0h.b[w14, 1]
// CHECK-ENCODING: [0x31,0x42,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0024231 <unknown>

movaz   z29.b, za0h.b[w12, 6]  // 11000000-00000010-00000010-11011101
// CHECK-INST: movaz   z29.b, za0h.b[w12, 6]
// CHECK-ENCODING: [0xdd,0x02,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00202dd <unknown>

movaz   z2.b, za0h.b[w15, 9]  // 11000000-00000010-01100011-00100010
// CHECK-INST: movaz   z2.b, za0h.b[w15, 9]
// CHECK-ENCODING: [0x22,0x63,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0026322 <unknown>

movaz   z7.b, za0h.b[w13, 12]  // 11000000-00000010-00100011-10000111
// CHECK-INST: movaz   z7.b, za0h.b[w13, 12]
// CHECK-ENCODING: [0x87,0x23,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0022387 <unknown>

movaz   z0.b, za0v.b[w12, 0]  // 11000000-00000010-10000010-00000000
// CHECK-INST: movaz   z0.b, za0v.b[w12, 0]
// CHECK-ENCODING: [0x00,0x82,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0028200 <unknown>

movaz   z21.b, za0v.b[w14, 10]  // 11000000-00000010-11000011-01010101
// CHECK-INST: movaz   z21.b, za0v.b[w14, 10]
// CHECK-ENCODING: [0x55,0xc3,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c002c355 <unknown>

movaz   z23.b, za0v.b[w15, 13]  // 11000000-00000010-11100011-10110111
// CHECK-INST: movaz   z23.b, za0v.b[w15, 13]
// CHECK-ENCODING: [0xb7,0xe3,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c002e3b7 <unknown>

movaz   z31.b, za0v.b[w15, 15]  // 11000000-00000010-11100011-11111111
// CHECK-INST: movaz   z31.b, za0v.b[w15, 15]
// CHECK-ENCODING: [0xff,0xe3,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c002e3ff <unknown>

movaz   z5.b, za0v.b[w12, 1]  // 11000000-00000010-10000010-00100101
// CHECK-INST: movaz   z5.b, za0v.b[w12, 1]
// CHECK-ENCODING: [0x25,0x82,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0028225 <unknown>

movaz   z1.b, za0v.b[w12, 1]  // 11000000-00000010-10000010-00100001
// CHECK-INST: movaz   z1.b, za0v.b[w12, 1]
// CHECK-ENCODING: [0x21,0x82,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0028221 <unknown>

movaz   z24.b, za0v.b[w14, 3]  // 11000000-00000010-11000010-01111000
// CHECK-INST: movaz   z24.b, za0v.b[w14, 3]
// CHECK-ENCODING: [0x78,0xc2,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c002c278 <unknown>

movaz   z0.b, za0v.b[w12, 12]  // 11000000-00000010-10000011-10000000
// CHECK-INST: movaz   z0.b, za0v.b[w12, 12]
// CHECK-ENCODING: [0x80,0x83,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c0028380 <unknown>

movaz   z17.b, za0v.b[w14, 1]  // 11000000-00000010-11000010-00110001
// CHECK-INST: movaz   z17.b, za0v.b[w14, 1]
// CHECK-ENCODING: [0x31,0xc2,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c002c231 <unknown>

movaz   z29.b, za0v.b[w12, 6]  // 11000000-00000010-10000010-11011101
// CHECK-INST: movaz   z29.b, za0v.b[w12, 6]
// CHECK-ENCODING: [0xdd,0x82,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c00282dd <unknown>

movaz   z2.b, za0v.b[w15, 9]  // 11000000-00000010-11100011-00100010
// CHECK-INST: movaz   z2.b, za0v.b[w15, 9]
// CHECK-ENCODING: [0x22,0xe3,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c002e322 <unknown>

movaz   z7.b, za0v.b[w13, 12]  // 11000000-00000010-10100011-10000111
// CHECK-INST: movaz   z7.b, za0v.b[w13, 12]
// CHECK-ENCODING: [0x87,0xa3,0x02,0xc0]
// CHECK-ERROR: instruction requires: sme2p1
// CHECK-UNKNOWN: c002a387 <unknown>
