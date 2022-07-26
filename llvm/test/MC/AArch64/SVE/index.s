// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme < %s \
// RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 \
// RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:        | llvm-objdump -d --mattr=+sve - | FileCheck %s --check-prefix=CHECK-INST
// RUN: llvm-mc -triple=aarch64 -filetype=obj -mattr=+sve < %s \
// RUN:   | llvm-objdump -d --mattr=-sve - | FileCheck %s --check-prefix=CHECK-UNKNOWN

// --------------------------------------------------------------------------//
// Index (immediate, immediate)

index   z0.b, #0, #0
// CHECK-INST: index   z0.b, #0, #0
// CHECK-ENCODING: [0x00,0x40,0x20,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04204000 <unknown>

index   z31.b, #-1, #-1
// CHECK-INST: index   z31.b, #-1, #-1
// CHECK-ENCODING: [0xff,0x43,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 043f43ff <unknown>

index   z0.h, #0, #0
// CHECK-INST: index   z0.h, #0, #0
// CHECK-ENCODING: [0x00,0x40,0x60,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04604000 <unknown>

index   z31.h, #-1, #-1
// CHECK-INST: index   z31.h, #-1, #-1
// CHECK-ENCODING: [0xff,0x43,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 047f43ff <unknown>

index   z0.s, #0, #0
// CHECK-INST: index   z0.s, #0, #0
// CHECK-ENCODING: [0x00,0x40,0xa0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a04000 <unknown>

index   z31.s, #-1, #-1
// CHECK-INST: index   z31.s, #-1, #-1
// CHECK-ENCODING: [0xff,0x43,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04bf43ff <unknown>

index   z0.d, #0, #0
// CHECK-INST: index   z0.d, #0, #0
// CHECK-ENCODING: [0x00,0x40,0xe0,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04e04000 <unknown>

index   z31.d, #-1, #-1
// CHECK-INST: index   z31.d, #-1, #-1
// CHECK-ENCODING: [0xff,0x43,0xff,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04ff43ff <unknown>

// --------------------------------------------------------------------------//
// Index (immediate, scalar)

index   z31.b, #-1, wzr
// CHECK-INST: index   z31.b, #-1, wzr
// CHECK-ENCODING: [0xff,0x4b,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 043f4bff <unknown>

index   z23.b, #13, w8
// CHECK-INST: index   z23.b, #13, w8
// CHECK-ENCODING: [0xb7,0x49,0x28,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 042849b7 <unknown>

index   z31.h, #-1, wzr
// CHECK-INST: index   z31.h, #-1, wzr
// CHECK-ENCODING: [0xff,0x4b,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 047f4bff <unknown>

index   z23.h, #13, w8
// CHECK-INST: index   z23.h, #13, w8
// CHECK-ENCODING: [0xb7,0x49,0x68,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 046849b7 <unknown>

index   z31.s, #-1, wzr
// CHECK-INST: index   z31.s, #-1, wzr
// CHECK-ENCODING: [0xff,0x4b,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04bf4bff <unknown>

index   z23.s, #13, w8
// CHECK-INST: index   z23.s, #13, w8
// CHECK-ENCODING: [0xb7,0x49,0xa8,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a849b7 <unknown>

index   z31.d, #-1, xzr
// CHECK-INST: index   z31.d, #-1, xzr
// CHECK-ENCODING: [0xff,0x4b,0xff,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04ff4bff <unknown>

index   z23.d, #13, x8
// CHECK-INST: index   z23.d, #13, x8
// CHECK-ENCODING: [0xb7,0x49,0xe8,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04e849b7 <unknown>


// --------------------------------------------------------------------------//
// Index (scalar, immediate)

index   z31.b, wzr, #-1
// CHECK-INST: index   z31.b, wzr, #-1
// CHECK-ENCODING: [0xff,0x47,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 043f47ff <unknown>

index   z23.b, w13, #8
// CHECK-INST: index   z23.b, w13, #8
// CHECK-ENCODING: [0xb7,0x45,0x28,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 042845b7 <unknown>

index   z31.h, wzr, #-1
// CHECK-INST: index   z31.h, wzr, #-1
// CHECK-ENCODING: [0xff,0x47,0x7f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 047f47ff <unknown>

index   z23.h, w13, #8
// CHECK-INST: index   z23.h, w13, #8
// CHECK-ENCODING: [0xb7,0x45,0x68,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 046845b7 <unknown>

index   z31.s, wzr, #-1
// CHECK-INST: index   z31.s, wzr, #-1
// CHECK-ENCODING: [0xff,0x47,0xbf,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04bf47ff <unknown>

index   z23.s, w13, #8
// CHECK-INST: index   z23.s, w13, #8
// CHECK-ENCODING: [0xb7,0x45,0xa8,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04a845b7 <unknown>

index   z31.d, xzr, #-1
// CHECK-INST: index   z31.d, xzr, #-1
// CHECK-ENCODING: [0xff,0x47,0xff,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04ff47ff <unknown>

index   z23.d, x13, #8
// CHECK-INST: index   z23.d, x13, #8
// CHECK-ENCODING: [0xb7,0x45,0xe8,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04e845b7 <unknown>


// --------------------------------------------------------------------------//
// Index (scalar, scalar)

index   z31.b, wzr, wzr
// CHECK-INST: index   z31.b, wzr, wzr
// CHECK-ENCODING: [0xff,0x4f,0x3f,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 043f4fff <unknown>

index   z21.b, w10, w21
// CHECK-INST: index   z21.b, w10, w21
// CHECK-ENCODING: [0x55,0x4d,0x35,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04354d55 <unknown>

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
// CHECK-UNKNOWN: 04bf4fff <unknown>

index   z21.s, w10, w21
// CHECK-INST: index   z21.s, w10, w21
// CHECK-ENCODING: [0x55,0x4d,0xb5,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04b54d55 <unknown>

index   z31.d, xzr, xzr
// CHECK-INST: index   z31.d, xzr, xzr
// CHECK-ENCODING: [0xff,0x4f,0xff,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04ff4fff <unknown>

index   z21.d, x10, x21
// CHECK-INST: index   z21.d, x10, x21
// CHECK-ENCODING: [0x55,0x4d,0xf5,0x04]
// CHECK-ERROR: instruction requires: sve or sme
// CHECK-UNKNOWN: 04f54d55 <unknown>
