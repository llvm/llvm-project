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

// --------------------------------------------------------------------------//
// Index (immediate, immediate)

index   z0.b, #0, #0
// CHECK-INST: index   z0.b, #0, #0
// CHECK-ENCODING: [0x00,0x40,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 40 20 04 <unknown>

index   z31.b, #-1, #-1
// CHECK-INST: index   z31.b, #-1, #-1
// CHECK-ENCODING: [0xff,0x43,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 43 3f 04 <unknown>

index   z0.h, #0, #0
// CHECK-INST: index   z0.h, #0, #0
// CHECK-ENCODING: [0x00,0x40,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 40 60 04 <unknown>

index   z31.h, #-1, #-1
// CHECK-INST: index   z31.h, #-1, #-1
// CHECK-ENCODING: [0xff,0x43,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 43 7f 04 <unknown>

index   z0.s, #0, #0
// CHECK-INST: index   z0.s, #0, #0
// CHECK-ENCODING: [0x00,0x40,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 40 a0 04 <unknown>

index   z31.s, #-1, #-1
// CHECK-INST: index   z31.s, #-1, #-1
// CHECK-ENCODING: [0xff,0x43,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 43 bf 04 <unknown>

index   z0.d, #0, #0
// CHECK-INST: index   z0.d, #0, #0
// CHECK-ENCODING: [0x00,0x40,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 00 40 e0 04 <unknown>

index   z31.d, #-1, #-1
// CHECK-INST: index   z31.d, #-1, #-1
// CHECK-ENCODING: [0xff,0x43,0xff,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 43 ff 04 <unknown>

// --------------------------------------------------------------------------//
// Index (immediate, scalar)

index   z31.b, #-1, wzr
// CHECK-INST: index   z31.b, #-1, wzr
// CHECK-ENCODING: [0xff,0x4b,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 4b 3f 04 <unknown>

index   z23.b, #13, w8
// CHECK-INST: index   z23.b, #13, w8
// CHECK-ENCODING: [0xb7,0x49,0x28,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 49 28 04 <unknown>

index   z31.h, #-1, wzr
// CHECK-INST: index   z31.h, #-1, wzr
// CHECK-ENCODING: [0xff,0x4b,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 4b 7f 04 <unknown>

index   z23.h, #13, w8
// CHECK-INST: index   z23.h, #13, w8
// CHECK-ENCODING: [0xb7,0x49,0x68,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 49 68 04 <unknown>

index   z31.s, #-1, wzr
// CHECK-INST: index   z31.s, #-1, wzr
// CHECK-ENCODING: [0xff,0x4b,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 4b bf 04 <unknown>

index   z23.s, #13, w8
// CHECK-INST: index   z23.s, #13, w8
// CHECK-ENCODING: [0xb7,0x49,0xa8,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 49 a8 04 <unknown>

index   z31.d, #-1, xzr
// CHECK-INST: index   z31.d, #-1, xzr
// CHECK-ENCODING: [0xff,0x4b,0xff,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 4b ff 04 <unknown>

index   z23.d, #13, x8
// CHECK-INST: index   z23.d, #13, x8
// CHECK-ENCODING: [0xb7,0x49,0xe8,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 49 e8 04 <unknown>


// --------------------------------------------------------------------------//
// Index (scalar, immediate)

index   z31.b, wzr, #-1
// CHECK-INST: index   z31.b, wzr, #-1
// CHECK-ENCODING: [0xff,0x47,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 47 3f 04 <unknown>

index   z23.b, w13, #8
// CHECK-INST: index   z23.b, w13, #8
// CHECK-ENCODING: [0xb7,0x45,0x28,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 45 28 04 <unknown>

index   z31.h, wzr, #-1
// CHECK-INST: index   z31.h, wzr, #-1
// CHECK-ENCODING: [0xff,0x47,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 47 7f 04 <unknown>

index   z23.h, w13, #8
// CHECK-INST: index   z23.h, w13, #8
// CHECK-ENCODING: [0xb7,0x45,0x68,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 45 68 04 <unknown>

index   z31.s, wzr, #-1
// CHECK-INST: index   z31.s, wzr, #-1
// CHECK-ENCODING: [0xff,0x47,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 47 bf 04 <unknown>

index   z23.s, w13, #8
// CHECK-INST: index   z23.s, w13, #8
// CHECK-ENCODING: [0xb7,0x45,0xa8,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 45 a8 04 <unknown>

index   z31.d, xzr, #-1
// CHECK-INST: index   z31.d, xzr, #-1
// CHECK-ENCODING: [0xff,0x47,0xff,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 47 ff 04 <unknown>

index   z23.d, x13, #8
// CHECK-INST: index   z23.d, x13, #8
// CHECK-ENCODING: [0xb7,0x45,0xe8,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: b7 45 e8 04 <unknown>


// --------------------------------------------------------------------------//
// Index (scalar, scalar)

index   z31.b, wzr, wzr
// CHECK-INST: index   z31.b, wzr, wzr
// CHECK-ENCODING: [0xff,0x4f,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 4f 3f 04 <unknown>

index   z21.b, w10, w21
// CHECK-INST: index   z21.b, w10, w21
// CHECK-ENCODING: [0x55,0x4d,0x35,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 4d 35 04 <unknown>

index   z31.h, wzr, wzr
// check-inst: index   z31.h, wzr, wzr
// check-encoding: [0xff,0x4f,0x7f,0x04]
// check-error: instruction requires: sve or sme
// check-unknown: ff 4f 7f 04 <unknown>

index   z0.h, w0, w0
// check-inst: index   z0.h, w0, w0
// check-encoding: [0x00,0x4c,0x60,0x04]
// check-error: instruction requires: sve or sme
// check-unknown: 00 4c 60 04 <unknown>

index   z31.s, wzr, wzr
// CHECK-INST: index   z31.s, wzr, wzr
// CHECK-ENCODING: [0xff,0x4f,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 4f bf 04 <unknown>

index   z21.s, w10, w21
// CHECK-INST: index   z21.s, w10, w21
// CHECK-ENCODING: [0x55,0x4d,0xb5,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 4d b5 04 <unknown>

index   z31.d, xzr, xzr
// CHECK-INST: index   z31.d, xzr, xzr
// CHECK-ENCODING: [0xff,0x4f,0xff,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: ff 4f ff 04 <unknown>

index   z21.d, x10, x21
// CHECK-INST: index   z21.d, x10, x21
// CHECK-ENCODING: [0x55,0x4d,0xf5,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 55 4d f5 04 <unknown>
