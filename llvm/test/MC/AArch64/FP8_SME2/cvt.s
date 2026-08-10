// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+fp8 < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+fp8 < %s \
// RUN:        | llvm-objdump -d --mattr=+sme2,+fp8 - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme2,+fp8 < %s \
// RUN:        | llvm-objdump -d --mattr=-sme2 - | FileCheck %s --check-prefix=CHECK-UNKNOWN
// Disassemble encoding and check the re-encoding (-show-encoding) matches.
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2,+fp8 < %s \
// RUN:        | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:        | llvm-mc -triple=aarch64 -mattr=+sme2,+fp8 -disassemble -show-encoding \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

f1cvt   {z0.h-z1.h}, z0.b  // 11000001-00100110-11100000-00000000
// CHECK-INST: f1cvt   { z0.h, z1.h }, z0.b
// CHECK-ENCODING: [0x00,0xe0,0x26,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c126e000 <unknown>

f1cvt   {z30.h-z31.h}, z31.b  // 11000001-00100110-11100011-11111110
// CHECK-INST: f1cvt   { z30.h, z31.h }, z31.b
// CHECK-ENCODING: [0xfe,0xe3,0x26,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c126e3fe <unknown>

f1cvtl  {z0.h-z1.h}, z0.b  // 11000001-00100110-11100000-00000001
// CHECK-INST: f1cvtl  { z0.h, z1.h }, z0.b
// CHECK-ENCODING: [0x01,0xe0,0x26,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c126e001 <unknown>

f1cvtl  {z30.h-z31.h}, z31.b  // 11000001-00100110-11100011-11111111
// CHECK-INST: f1cvtl  { z30.h, z31.h }, z31.b
// CHECK-ENCODING: [0xff,0xe3,0x26,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c126e3ff <unknown>

bf1cvt  {z0.h-z1.h}, z0.b  // 11000001-01100110-11100000-00000000
// CHECK-INST: bf1cvt  { z0.h, z1.h }, z0.b
// CHECK-ENCODING: [0x00,0xe0,0x66,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c166e000 <unknown>

bf1cvt  {z30.h-z31.h}, z31.b  // 11000001-01100110-11100011-11111110
// CHECK-INST: bf1cvt  { z30.h, z31.h }, z31.b
// CHECK-ENCODING: [0xfe,0xe3,0x66,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c166e3fe <unknown>

bf1cvtl {z0.h-z1.h}, z0.b  // 11000001-01100110-11100000-00000001
// CHECK-INST: bf1cvtl { z0.h, z1.h }, z0.b
// CHECK-ENCODING: [0x01,0xe0,0x66,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c166e001 <unknown>

bf1cvtl {z30.h-z31.h}, z31.b  // 11000001-01100110-11100011-11111111
// CHECK-INST: bf1cvtl { z30.h, z31.h }, z31.b
// CHECK-ENCODING: [0xff,0xe3,0x66,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c166e3ff <unknown>

bf2cvt  {z0.h-z1.h}, z0.b  // 11000001-11100110-11100000-00000000
// CHECK-INST: bf2cvt  { z0.h, z1.h }, z0.b
// CHECK-ENCODING: [0x00,0xe0,0xe6,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1e6e000 <unknown>

bf2cvt  {z30.h-z31.h}, z31.b  // 11000001-11100110-11100011-11111110
// CHECK-INST: bf2cvt  { z30.h, z31.h }, z31.b
// CHECK-ENCODING: [0xfe,0xe3,0xe6,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1e6e3fe <unknown>

bf2cvtl {z0.h-z1.h}, z0.b  // 11000001-11100110-11100000-00000001
// CHECK-INST: bf2cvtl { z0.h, z1.h }, z0.b
// CHECK-ENCODING: [0x01,0xe0,0xe6,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1e6e001 <unknown>

bf2cvtl {z30.h-z31.h}, z31.b  // 11000001-11100110-11100011-11111111
// CHECK-INST: bf2cvtl { z30.h, z31.h }, z31.b
// CHECK-ENCODING: [0xff,0xe3,0xe6,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1e6e3ff <unknown>

f2cvt   {z0.h-z1.h}, z0.b  // 11000001-10100110-11100000-00000000
// CHECK-INST: f2cvt   { z0.h, z1.h }, z0.b
// CHECK-ENCODING: [0x00,0xe0,0xa6,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1a6e000 <unknown>

f2cvt   {z30.h-z31.h}, z31.b  // 11000001-10100110-11100011-11111110
// CHECK-INST: f2cvt   { z30.h, z31.h }, z31.b
// CHECK-ENCODING: [0xfe,0xe3,0xa6,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1a6e3fe <unknown>

f2cvtl  {z0.h-z1.h}, z0.b  // 11000001-10100110-11100000-00000001
// CHECK-INST: f2cvtl  { z0.h, z1.h }, z0.b
// CHECK-ENCODING: [0x01,0xe0,0xa6,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1a6e001 <unknown>

f2cvtl  {z30.h-z31.h}, z31.b  // 11000001-10100110-11100011-11111111
// CHECK-INST: f2cvtl  { z30.h, z31.h }, z31.b
// CHECK-ENCODING: [0xff,0xe3,0xa6,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c1a6e3ff <unknown>

fcvt    z0.b, {z0.h-z1.h}  // 11000001-00100100-11100000-00000000
// CHECK-INST: fcvt    z0.b, { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0xe0,0x24,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c124e000 <unknown>

fcvt    z31.b, {z30.h-z31.h}  // 11000001-00100100-11100011-11011111
// CHECK-INST: fcvt    z31.b, { z30.h, z31.h }
// CHECK-ENCODING: [0xdf,0xe3,0x24,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c124e3df <unknown>

fcvt    z0.b, {z0.s-z3.s}  // 11000001-00110100-11100000-00000000
// CHECK-INST: fcvt    z0.b, { z0.s - z3.s }
// CHECK-ENCODING: [0x00,0xe0,0x34,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c134e000 <unknown>

fcvt    z31.b, {z28.s-z31.s}  // 11000001-00110100-11100011-10011111
// CHECK-INST: fcvt    z31.b, { z28.s - z31.s }
// CHECK-ENCODING: [0x9f,0xe3,0x34,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c134e39f <unknown>

fcvtn   z0.b, {z0.s-z3.s}  // 11000001-00110100-11100000-00100000
// CHECK-INST: fcvtn   z0.b, { z0.s - z3.s }
// CHECK-ENCODING: [0x20,0xe0,0x34,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c134e020 <unknown>

fcvtn   z31.b, {z28.s-z31.s}  // 11000001-00110100-11100011-10111111
// CHECK-INST: fcvtn   z31.b, { z28.s - z31.s }
// CHECK-ENCODING: [0xbf,0xe3,0x34,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c134e3bf <unknown>

bfcvt   z0.b, {z0.h-z1.h}  // 11000001-01100100-11100000-00000000
// CHECK-INST: bfcvt   z0.b, { z0.h, z1.h }
// CHECK-ENCODING: [0x00,0xe0,0x64,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c164e000 <unknown>

bfcvt   z31.b, {z30.h-z31.h}  // 11000001-01100100-11100011-11011111
// CHECK-INST: bfcvt   z31.b, { z30.h, z31.h }
// CHECK-ENCODING: [0xdf,0xe3,0x64,0xc1]
// CHECK-ERROR: instruction requires: fp8 sme2
// CHECK-UNKNOWN: c164e3df <unknown>
