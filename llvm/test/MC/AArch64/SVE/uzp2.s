// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+streaming-sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

uzp2    z31.b, z31.b, z31.b
// CHECK-INST: uzp2	z31.b, z31.b, z31.b
// CHECK-ENCODING: [0xff,0x6f,0x3f,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 6f 3f 05 <unknown>

uzp2    z31.h, z31.h, z31.h
// CHECK-INST: uzp2	z31.h, z31.h, z31.h
// CHECK-ENCODING: [0xff,0x6f,0x7f,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 6f 7f 05 <unknown>

uzp2    z31.s, z31.s, z31.s
// CHECK-INST: uzp2	z31.s, z31.s, z31.s
// CHECK-ENCODING: [0xff,0x6f,0xbf,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 6f bf 05 <unknown>

uzp2    z31.d, z31.d, z31.d
// CHECK-INST: uzp2	z31.d, z31.d, z31.d
// CHECK-ENCODING: [0xff,0x6f,0xff,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ff 6f ff 05 <unknown>

uzp2    p15.b, p15.b, p15.b
// CHECK-INST: uzp2	p15.b, p15.b, p15.b
// CHECK-ENCODING: [0xef,0x4d,0x2f,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ef 4d 2f 05 <unknown>

uzp2    p15.s, p15.s, p15.s
// CHECK-INST: uzp2	p15.s, p15.s, p15.s
// CHECK-ENCODING: [0xef,0x4d,0xaf,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ef 4d af 05 <unknown>

uzp2    p15.h, p15.h, p15.h
// CHECK-INST: uzp2	p15.h, p15.h, p15.h
// CHECK-ENCODING: [0xef,0x4d,0x6f,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ef 4d 6f 05 <unknown>

uzp2    p15.d, p15.d, p15.d
// CHECK-INST: uzp2	p15.d, p15.d, p15.d
// CHECK-ENCODING: [0xef,0x4d,0xef,0x05]
// CHECK-ERROR: instruction requires: streaming-sve or sve
// CHECK-UNKNOWN: ef 4d ef 05 <unknown>
