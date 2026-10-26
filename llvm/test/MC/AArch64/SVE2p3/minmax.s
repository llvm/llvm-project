// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3,+fpv2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2p3,+fpv2 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p3,+fpv2 < %s \
// RUN:        | llvm-objdump -d --mattr=+sve2p3,+fpv2 --no-print-imm-hex - \
// RUN:        | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve2p3,+fpv2 < %s \
// RUN:        | llvm-objdump -d --mattr=-all --no-print-imm-hex - \
// RUN:        | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p3,+fpv2 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sve2p3,+fpv2 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

// FMAXINM (predicated vectors)

fmaxinm z0.h, p0/m, z0.h, z0.h
// CHECK-INST: fmaxinm z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: encoding: [0x00,0xe0,0x40,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6440e000 <unknown>

fmaxinm z31.h, p7/m, z31.h, z31.h
// CHECK-INST: fmaxinm z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: encoding: [0xff,0xff,0x40,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6440ffff <unknown>

fmaxinm z0.s, p0/m, z0.s, z0.s
// CHECK-INST: fmaxinm z0.s, p0/m, z0.s, z0.s
// CHECK-ENCODING: encoding: [0x00,0xe0,0x80,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6480e000 <unknown>

fmaxinm z31.s, p7/m, z31.s, z31.s
// CHECK-INST: fmaxinm z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: encoding: [0xff,0xff,0x80,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6480ffff <unknown>

fmaxinm z0.d, p0/m, z0.d, z0.d
// CHECK-INST: fmaxinm z0.d, p0/m, z0.d, z0.d
// CHECK-ENCODING: encoding: [0x00,0xe0,0xc0,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64c0e000 <unknown>

fmaxinm z31.d, p7/m, z31.d, z31.d
// CHECK-INST: fmaxinm z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: encoding: [0xff,0xff,0xc0,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64c0ffff <unknown>

movprfx  z0.h, p0/m, z27.h
fmaxinm z0.h, p0/m, z0.h, z31.h
// CHECK-INST: fmaxinm z0.h, p0/m, z0.h, z31.h
// CHECK-ENCODING: encoding: [0xe0,0xe3,0x40,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6440e3e0 <unknown>

// FMININM (predicated vectors)

fmininm z0.h, p0/m, z0.h, z0.h
// CHECK-INST: fmininm z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: encoding: [0x00,0xe0,0x41,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6441e000 <unknown>

fmininm z31.h, p7/m, z31.h, z31.h
// CHECK-INST: fmininm z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: encoding: [0xff,0xff,0x41,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6441ffff <unknown>

fmininm z0.s, p0/m, z0.s, z0.s
// CHECK-INST: fmininm z0.s, p0/m, z0.s, z0.s
// CHECK-ENCODING: encoding: [0x00,0xe0,0x81,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6481e000 <unknown>

fmininm z31.s, p7/m, z31.s, z31.s
// CHECK-INST: fmininm z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: encoding: [0xff,0xff,0x81,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6481ffff <unknown>

fmininm z0.d, p0/m, z0.d, z0.d
// CHECK-INST: fmininm z0.d, p0/m, z0.d, z0.d
// CHECK-ENCODING: encoding: [0x00,0xe0,0xc1,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64c1e000 <unknown>

fmininm z31.d, p7/m, z31.d, z31.d
// CHECK-INST: fmininm z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: encoding: [0xff,0xff,0xc1,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64c1ffff <unknown>

movprfx  z0.h, p0/m, z27.h
fmininm z0.h, p0/m, z0.h, z31.h
// CHECK-INST: fmininm z0.h, p0/m, z0.h, z31.h
// CHECK-ENCODING: encoding: [0xe0,0xe3,0x41,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6441e3e0 <unknown>

// FMAXMG (predicated vectors)

fmaxmg z0.h, p0/m, z0.h, z0.h
// CHECK-INST: fmaxmg z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: encoding: [0x00,0xe0,0x4a,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 644ae000 <unknown>

fmaxmg z31.h, p7/m, z31.h, z31.h
// CHECK-INST: fmaxmg z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: encoding: [0xff,0xff,0x4a,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 644affff <unknown>

fmaxmg z0.s, p0/m, z0.s, z0.s
// CHECK-INST: fmaxmg z0.s, p0/m, z0.s, z0.s
// CHECK-ENCODING: encoding: [0x00,0xe0,0x8a,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 648ae000 <unknown>

fmaxmg z31.s, p7/m, z31.s, z31.s
// CHECK-INST: fmaxmg z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: encoding: [0xff,0xff,0x8a,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 648affff <unknown>

fmaxmg z0.d, p0/m, z0.d, z0.d
// CHECK-INST: fmaxmg z0.d, p0/m, z0.d, z0.d
// CHECK-ENCODING: encoding: [0x00,0xe0,0xca,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64cae000 <unknown>

fmaxmg z31.d, p7/m, z31.d, z31.d
// CHECK-INST: fmaxmg z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: encoding: [0xff,0xff,0xca,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64caffff <unknown>

movprfx  z0.h, p0/m, z27.h
fmaxmg z0.h, p0/m, z0.h, z31.h
// CHECK-INST: fmaxmg z0.h, p0/m, z0.h, z31.h
// CHECK-ENCODING: encoding: [0xe0,0xe3,0x4a,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 644ae3e0 <unknown>

// FMINMG (predicated vectors)

fminmg z0.h, p0/m, z0.h, z0.h
// CHECK-INST: fminmg z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: encoding: [0x00,0xe0,0x4b,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 644be000 <unknown>

fminmg z31.h, p7/m, z31.h, z31.h
// CHECK-INST: fminmg z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: encoding: [0xff,0xff,0x4b,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 644bffff <unknown>

fminmg z0.s, p0/m, z0.s, z0.s
// CHECK-INST: fminmg z0.s, p0/m, z0.s, z0.s
// CHECK-ENCODING: encoding: [0x00,0xe0,0x8b,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 648be000 <unknown>

fminmg z31.s, p7/m, z31.s, z31.s
// CHECK-INST: fminmg z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: encoding: [0xff,0xff,0x8b,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 648bffff <unknown>

fminmg z0.d, p0/m, z0.d, z0.d
// CHECK-INST: fminmg z0.d, p0/m, z0.d, z0.d
// CHECK-ENCODING: encoding: [0x00,0xe0,0xcb,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64cbe000 <unknown>

fminmg z31.d, p7/m, z31.d, z31.d
// CHECK-INST: fminmg z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: encoding: [0xff,0xff,0xcb,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64cbffff <unknown>

movprfx  z0.h, p0/m, z27.h
fminmg z0.h, p0/m, z0.h, z31.h
// CHECK-INST: fminmg z0.h, p0/m, z0.h, z31.h
// CHECK-ENCODING: encoding: [0xe0,0xe3,0x4b,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 644be3e0 <unknown>

// FMAXMGINM (predicated vectors)

fmaxmginm z0.h, p0/m, z0.h, z0.h
// CHECK-INST: fmaxmginm z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: encoding: [0x00,0xe0,0x48,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6448e000 <unknown>

fmaxmginm z31.h, p7/m, z31.h, z31.h
// CHECK-INST: fmaxmginm z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: encoding: [0xff,0xff,0x48,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6448ffff <unknown>

fmaxmginm z0.s, p0/m, z0.s, z0.s
// CHECK-INST: fmaxmginm z0.s, p0/m, z0.s, z0.s
// CHECK-ENCODING: encoding: [0x00,0xe0,0x88,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6488e000 <unknown>

fmaxmginm z31.s, p7/m, z31.s, z31.s
// CHECK-INST: fmaxmginm z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: encoding: [0xff,0xff,0x88,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6488ffff <unknown>

fmaxmginm z0.d, p0/m, z0.d, z0.d
// CHECK-INST: fmaxmginm z0.d, p0/m, z0.d, z0.d
// CHECK-ENCODING: encoding: [0x00,0xe0,0xc8,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64c8e000 <unknown>

fmaxmginm z31.d, p7/m, z31.d, z31.d
// CHECK-INST: fmaxmginm z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: encoding: [0xff,0xff,0xc8,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64c8ffff <unknown>

movprfx  z0.h, p0/m, z27.h
fmaxmginm z0.h, p0/m, z0.h, z31.h
// CHECK-INST: fmaxmginm z0.h, p0/m, z0.h, z31.h
// CHECK-ENCODING: encoding: [0xe0,0xe3,0x48,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6448e3e0 <unknown>

// FMINMGINM (predicated vectors)

fminmginm z0.h, p0/m, z0.h, z0.h
// CHECK-INST: fminmginm z0.h, p0/m, z0.h, z0.h
// CHECK-ENCODING: encoding: [0x00,0xe0,0x49,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6449e000 <unknown>

fminmginm z31.h, p7/m, z31.h, z31.h
// CHECK-INST: fminmginm z31.h, p7/m, z31.h, z31.h
// CHECK-ENCODING: encoding: [0xff,0xff,0x49,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6449ffff <unknown>

fminmginm z0.s, p0/m, z0.s, z0.s
// CHECK-INST: fminmginm z0.s, p0/m, z0.s, z0.s
// CHECK-ENCODING: encoding: [0x00,0xe0,0x89,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6489e000 <unknown>

fminmginm z31.s, p7/m, z31.s, z31.s
// CHECK-INST: fminmginm z31.s, p7/m, z31.s, z31.s
// CHECK-ENCODING: encoding: [0xff,0xff,0x89,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6489ffff <unknown>

fminmginm z0.d, p0/m, z0.d, z0.d
// CHECK-INST: fminmginm z0.d, p0/m, z0.d, z0.d
// CHECK-ENCODING: encoding: [0x00,0xe0,0xc9,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64c9e000 <unknown>

fminmginm z31.d, p7/m, z31.d, z31.d
// CHECK-INST: fminmginm z31.d, p7/m, z31.d, z31.d
// CHECK-ENCODING: encoding: [0xff,0xff,0xc9,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64c9ffff <unknown>

movprfx  z0.h, p0/m, z27.h
fminmginm z0.h, p0/m, z0.h, z31.h
// CHECK-INST: fminmginm z0.h, p0/m, z0.h, z31.h
// CHECK-ENCODING: encoding: [0xe0,0xe3,0x49,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6449e3e0 <unknown>

// FMAXINMV (reductions)

fmaxinmv h0, p0, z0.h
// CHECK-INST: fmaxinmv h0, p0, z0.h
// CHECK-ENCODING: encoding: [0x00,0xc0,0x54,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6454c000 <unknown>

fmaxinmv h31, p7, z31.h
// CHECK-INST: fmaxinmv h31, p7, z31.h
// CHECK-ENCODING: encoding: [0xff,0xdf,0x54,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6454dfff <unknown>

fmaxinmv s0, p0, z0.s
// CHECK-INST: fmaxinmv s0, p0, z0.s
// CHECK-ENCODING: encoding: [0x00,0xc0,0x94,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6494c000 <unknown>

fmaxinmv s31, p7, z31.s
// CHECK-INST: fmaxinmv s31, p7, z31.s
// CHECK-ENCODING: encoding: [0xff,0xdf,0x94,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6494dfff <unknown>

fmaxinmv d0, p0, z0.d
// CHECK-INST: fmaxinmv d0, p0, z0.d
// CHECK-ENCODING: encoding: [0x00,0xc0,0xd4,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64d4c000 <unknown>

fmaxinmv d31, p7, z31.d
// CHECK-INST: fmaxinmv d31, p7, z31.d
// CHECK-ENCODING: encoding: [0xff,0xdf,0xd4,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64d4dfff <unknown>

// FMININMV (reductions)

fmininmv h0, p0, z0.h
// CHECK-INST: fmininmv h0, p0, z0.h
// CHECK-ENCODING: encoding: [0x00,0xc0,0x55,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6455c000 <unknown>

fmininmv h31, p7, z31.h
// CHECK-INST: fmininmv h31, p7, z31.h
// CHECK-ENCODING: encoding: [0xff,0xdf,0x55,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6455dfff <unknown>

fmininmv s0, p0, z0.s
// CHECK-INST: fmininmv s0, p0, z0.s
// CHECK-ENCODING: encoding: [0x00,0xc0,0x95,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6495c000 <unknown>

fmininmv s31, p7, z31.s
// CHECK-INST: fmininmv s31, p7, z31.s
// CHECK-ENCODING: encoding: [0xff,0xdf,0x95,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6495dfff <unknown>

fmininmv d0, p0, z0.d
// CHECK-INST: fmininmv d0, p0, z0.d
// CHECK-ENCODING: encoding: [0x00,0xc0,0xd5,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64d5c000 <unknown>

fmininmv d31, p7, z31.d
// CHECK-INST: fmininmv d31, p7, z31.d
// CHECK-ENCODING: encoding: [0xff,0xdf,0xd5,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64d5dfff <unknown>

// FMAXMGV (reductions)

fmaxmgv h0, p0, z0.h
// CHECK-INST: fmaxmgv h0, p0, z0.h
// CHECK-ENCODING: encoding: [0x00,0xe0,0x56,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6456e000 <unknown>

fmaxmgv h31, p7, z31.h
// CHECK-INST: fmaxmgv h31, p7, z31.h
// CHECK-ENCODING: encoding: [0xff,0xff,0x56,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6456ffff <unknown>

fmaxmgv s0, p0, z0.s
// CHECK-INST: fmaxmgv s0, p0, z0.s
// CHECK-ENCODING: encoding: [0x00,0xe0,0x96,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6496e000 <unknown>

fmaxmgv s31, p7, z31.s
// CHECK-INST: fmaxmgv s31, p7, z31.s
// CHECK-ENCODING: encoding: [0xff,0xff,0x96,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6496ffff <unknown>

fmaxmgv d0, p0, z0.d
// CHECK-INST: fmaxmgv d0, p0, z0.d
// CHECK-ENCODING: encoding: [0x00,0xe0,0xd6,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64d6e000 <unknown>

fmaxmgv d31, p7, z31.d
// CHECK-INST: fmaxmgv d31, p7, z31.d
// CHECK-ENCODING: encoding: [0xff,0xff,0xd6,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64d6ffff <unknown>

// FMINMGV (reductions)

fminmgv h0, p0, z0.h
// CHECK-INST: fminmgv h0, p0, z0.h
// CHECK-ENCODING: encoding: [0x00,0xe0,0x57,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6457e000 <unknown>

fminmgv h31, p7, z31.h
// CHECK-INST: fminmgv h31, p7, z31.h
// CHECK-ENCODING: encoding: [0xff,0xff,0x57,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6457ffff <unknown>

fminmgv s0, p0, z0.s
// CHECK-INST: fminmgv s0, p0, z0.s
// CHECK-ENCODING: encoding: [0x00,0xe0,0x97,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6497e000 <unknown>

fminmgv s31, p7, z31.s
// CHECK-INST: fminmgv s31, p7, z31.s
// CHECK-ENCODING: encoding: [0xff,0xff,0x97,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6497ffff <unknown>

fminmgv d0, p0, z0.d
// CHECK-INST: fminmgv d0, p0, z0.d
// CHECK-ENCODING: encoding: [0x00,0xe0,0xd7,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64d7e000 <unknown>

fminmgv d31, p7, z31.d
// CHECK-INST: fminmgv d31, p7, z31.d
// CHECK-ENCODING: encoding: [0xff,0xff,0xd7,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64d7ffff <unknown>

// FMAXMGINMV (reductions)

fmaxmginmv h0, p0, z0.h
// CHECK-INST: fmaxmginmv h0, p0, z0.h
// CHECK-ENCODING: encoding: [0x00,0xe0,0x54,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6454e000 <unknown>

fmaxmginmv h31, p7, z31.h
// CHECK-INST: fmaxmginmv h31, p7, z31.h
// CHECK-ENCODING: encoding: [0xff,0xff,0x54,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6454ffff <unknown>

fmaxmginmv s0, p0, z0.s
// CHECK-INST: fmaxmginmv s0, p0, z0.s
// CHECK-ENCODING: encoding: [0x00,0xe0,0x94,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6494e000 <unknown>

fmaxmginmv s31, p7, z31.s
// CHECK-INST: fmaxmginmv s31, p7, z31.s
// CHECK-ENCODING: encoding: [0xff,0xff,0x94,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6494ffff <unknown>

fmaxmginmv d0, p0, z0.d
// CHECK-INST: fmaxmginmv d0, p0, z0.d
// CHECK-ENCODING: encoding: [0x00,0xe0,0xd4,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64d4e000 <unknown>

fmaxmginmv d31, p7, z31.d
// CHECK-INST: fmaxmginmv d31, p7, z31.d
// CHECK-ENCODING: encoding: [0xff,0xff,0xd4,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64d4ffff <unknown>

// FMINMGINMV (reductions)

fminmginmv h0, p0, z0.h
// CHECK-INST: fminmginmv h0, p0, z0.h
// CHECK-ENCODING: encoding: [0x00,0xe0,0x55,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6455e000 <unknown>

fminmginmv h31, p7, z31.h
// CHECK-INST: fminmginmv h31, p7, z31.h
// CHECK-ENCODING: encoding: [0xff,0xff,0x55,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6455ffff <unknown>

fminmginmv s0, p0, z0.s
// CHECK-INST: fminmginmv s0, p0, z0.s
// CHECK-ENCODING: encoding: [0x00,0xe0,0x95,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6495e000 <unknown>

fminmginmv s31, p7, z31.s
// CHECK-INST: fminmginmv s31, p7, z31.s
// CHECK-ENCODING: encoding: [0xff,0xff,0x95,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 6495ffff <unknown>

fminmginmv d0, p0, z0.d
// CHECK-INST: fminmginmv d0, p0, z0.d
// CHECK-ENCODING: encoding: [0x00,0xe0,0xd5,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64d5e000 <unknown>

fminmginmv d31, p7, z31.d
// CHECK-INST: fminmginmv d31, p7, z31.d
// CHECK-ENCODING: encoding: [0xff,0xff,0xd5,0x64]
// CHECK-ERROR: instruction requires: fpv2 sme2p3 or sve2p3
// CHECK-UNKNOWN: 64d5ffff <unknown>
