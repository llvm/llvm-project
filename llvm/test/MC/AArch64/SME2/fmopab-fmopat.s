// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-f16f8f16 < %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-f16f8f16 < %s \
// RUN:   | llvm-objdump -d --mattr=+sme-f16f8f16 --no-print-imm-hex - \
// RUN:   | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sme-f16f8f16 < %s \
// RUN:   | llvm-objdump -d --mattr=-sme-f16f8f16 --no-print-imm-hex - \
// RUN:   | FileCheck %s --check-prefix=CHECK-UNKNOWN
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme-f16f8f16 < %s \
// RUN:   | sed '/.text/d' | sed 's/.*encoding: //g' \
// RUN:   | llvm-mc -triple=aarch64 -mattr=+sme-f16f8f16 -disassemble -show-encoding \
// RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

fmopab za0.h, { z0.h, z1.h }, z0.b, z20.b
// CHECK-INST: fmopab za0.h, { z0.h, z1.h }, z0.b, z20.b
// CHECK-ENCODING: encoding: [0x18,0x00,0xa0,0x80]
// CHECK-ERROR: instruction requires: sme-f16f8f16
// CHECK-UNKNOWN: 80a00018 <unknown>

fmopab za1.h, { z30.h, z31.h }, z0.b, z23.b
// CHECK-INST: fmopab za1.h, { z30.h, z31.h }, z0.b, z23.b
// CHECK-ENCODING: encoding: [0xd9,0x0f,0xa0,0x80]
// CHECK-ERROR: instruction requires: sme-f16f8f16
// CHECK-UNKNOWN: 80a00fd9 <unknown>

fmopat za0.h, { z0.h, z1.h }, z31.b, z28.b
// CHECK-INST: fmopat za0.h, { z0.h, z1.h }, z31.b, z28.b
// CHECK-ENCODING: encoding: [0x18,0x90,0xbf,0x80]
// CHECK-ERROR: instruction requires: sme-f16f8f16
// CHECK-UNKNOWN: 80bf9018 <unknown>

fmopat za1.h, { z30.h, z31.h }, z31.b, z31.b
// CHECK-INST: fmopat za1.h, { z30.h, z31.h }, z31.b, z31.b
// CHECK-ENCODING: encoding: [0xd9,0x9f,0xbf,0x80]
// CHECK-ERROR: instruction requires: sme-f16f8f16
// CHECK-UNKNOWN: 80bf9fd9 <unknown>
