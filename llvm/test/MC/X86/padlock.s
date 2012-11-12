// RUN: llvm-mc -triple i386-unknown-unknown --show-encoding %s | FileCheck %s

	xstore
// CHECK: xstore
// CHECK: encoding: [0x0f,0xa7,0xc0]

	xstorerng
// CHECK: xstore
// CHECK: encoding: [0x0f,0xa7,0xc0]

	rep xcryptecb
// CHECK: rep
// CHECK: encoding: [0xf3]
// CHECK: xcryptecb
// CHECK: encoding: [0x0f,0xa7,0xc8]

	rep xcryptcbc
// CHECK: rep
// CHECK: encoding: [0xf3]
// CHECK: xcryptcbc
// CHECK: encoding: [0x0f,0xa7,0xd0]

	rep xcryptctr
// CHECK: rep
// CHECK: encoding: [0xf3]
// CHECK: xcryptctr
// CHECK: encoding: [0x0f,0xa7,0xd8]

	rep xcryptcfb
// CHECK: rep
// CHECK: encoding: [0xf3]
// CHECK: xcryptcfb
// CHECK: encoding: [0x0f,0xa7,0xe0]

	rep xcryptofb
// CHECK: rep
// CHECK: encoding: [0xf3]
// CHECK: xcryptofb
// CHECK: encoding: [0x0f,0xa7,0xe8]

	rep xsha1
// CHECK: rep
// CHECK: encoding: [0xf3]
// CHECK: xsha1
// CHECK: encoding: [0x0f,0xa6,0xc8]

	rep xsha256
// CHECK: rep
// CHECK: encoding: [0xf3]
// CHECK: xsha256
// CHECK: encoding: [0x0f,0xa6,0xd0]

	rep montmul
// CHECK: rep
// CHECK: encoding: [0xf3]
// CHECK: montmul
// CHECK: encoding: [0x0f,0xa6,0xc0]
