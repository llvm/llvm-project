# RUN: llvm-mc -triple=riscv64 -show-encoding -mcpu=spacemit-a100 %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENC

# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s

# RUN: llvm-mc -triple=riscv64 -filetype=obj -mcpu=spacemit-a100 %s \
# RUN:        | llvm-objdump -d  --mcpu=spacemit-a100 -M no-aliases - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

// CHECK-ENC:   encoding: [0x2b,0xb0,0x2f,0xc2]
// CHECK-INST:  smt.vmadot	v0, v31, v2, i4
smt.vmadot v0, v31, v2, i4 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0x2b,0xb0,0x2f,0xe2]
// CHECK-INST:  smt.vmadot	v0, v31, v2, i8
smt.vmadot v0, v31, v2, i8 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0x2b,0xb0,0x2f,0xe2]
// CHECK-INST:  smt.vmadot	v0, v31, v2, i8
smt.vmadot v0, v31, v2
// expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
// expected-note@-2:1 {{instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension)}}
// expected-note@-3:1 {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0x2b,0x30,0x41,0xe6]
// CHECK-INST:  smt.vmadot1	v0, v2, v4, i8
smt.vmadot1 v0, v2, v4, i8 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0x2b,0x30,0x41,0xe6]
// CHECK-INST:  smt.vmadot1	v0, v2, v4, i8
smt.vmadot1 v0, v2, v4, i8 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0x2b,0x30,0x41,0xe6]
// CHECK-INST:  smt.vmadot1	v0, v2, v4, i8
smt.vmadot1 v0, v2, v4
// expected-error@-1:1 {{invalid instruction, any one of the following would fix this:}}
// expected-note@-2:1 {{instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension)}}
// expected-note@-3:1 {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0xab,0x38,0x81,0xc8]
// CHECK-INST:  smt.vmadot.sp	v16, v2, v8, v0, 0x1, i4
smt.vmadot.sp v16, v2, v8, v0, 1, i4 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0xab,0x38,0x81,0xe8]
// CHECK-INST:  smt.vmadot.sp	v16, v2, v8, v0, 0x1, i8
smt.vmadot.sp v16, v2, v8, v0, 1, i8 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0xab,0x38,0x81,0xe8]
// CHECK-INST:  smt.vmadot.sp	v16, v2, v8, v0, 0x1, i8
smt.vmadot.sp v16, v2, v8, v0, 1 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0x2b,0xb8,0x81,0xea]
// CHECK-INST:  smt.vmadot.sp	v16, v2, v8, v1, 0x2, i8
smt.vmadot.sp v16, v2, v8, v1, 2, i8 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC: smt.vmadot.sp	v16, v2, v1, v0, 1, i4  # encoding: [0xab,0x38,0x11,0xc8]
// CHECK-INST:	smt.vmadot.sp	v16, v2, v1, v0, 0x1, i4
// CHECK-UNKNOWN: c81138ab     	<unknown>
smt.vmadot.sp v16, v2, v1, v0, 1, i4 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0x2b,0x38,0x81,0xd0]
// CHECK-INST:  smt.vmadot.hp	v16, v2, v8, v0, 0x3, i4
smt.vmadot.hp v16, v2, v8, v0, 3, i4 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0x2b,0x28,0x81,0xf0]
// CHECK-INST:  smt.vmadot.hp	v16, v2, v8, v0, 0x2, i8
smt.vmadot.hp v16, v2, v8, v0, 2, i8 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0x2b,0x28,0x81,0xf0]
// CHECK-INST:  smt.vmadot.hp	v16, v2, v8, v0, 0x2, i8
smt.vmadot.hp v16, v2, v8, v0, 2 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0x2b,0x48,0x81,0xf2]
// CHECK-INST:  smt.vmadot.hp v16, v2, v8, v1, 0x4, i8
smt.vmadot.hp v16, v2, v8, v1, 4, i8 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0x2b,0x48,0x81,0xf2]
// CHECK-INST:  smt.vmadot.hp	v16, v2, v8, v1, 0x4, i8
smt.vmadot.hp v16, v2, v8, v1, 4 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC: smt.vmadot.hp	v1, v2, v8, v0, 3, i4   # encoding: [0xab,0x30,0x81,0xd0]
// CHECK-INST: 	smt.vmadot.hp	v1, v2, v8, v0, 0x3, i4
// CHECK-UNKNOWN: d08130ab     	<unknown>
smt.vmadot.hp v1, v2, v8, v0, 3, i4 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC: smt.vmadot.hp	v16, v1, v8, v0, 3, i4  # encoding: [0x2b,0xb8,0x80,0xd0]
// CHECK-INST: 	smt.vmadot.hp	v16, v1, v8, v0, 0x3, i4
// CHECK-UNKNOWN: d080b82b     	<unknown>
smt.vmadot.hp v16, v1, v8, v0, 3, i4 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC: smt.vmadot.hp	v16, v2, v1, v0, 3, i4  # encoding: [0x2b,0x38,0x11,0xd0]
// CHECK-INST: 	smt.vmadot.hp	v16, v2, v1, v0, 0x3, i4
// CHECK-UNKNOWN: d011382b     	<unknown>
smt.vmadot.hp v16, v2, v1, v0, 3, i4 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0x2b,0x41,0x62,0x9e]
// CHECK-INST:  smt.vfwmadot	v2, v4, v6
smt.vfwmadot v2, v4, v6 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0x2b,0x51,0x62,0x9e]
// CHECK-INST:  smt.vfwmadot1	v2, v4, v6
smt.vfwmadot1 v2, v4, v6 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0xab,0x90,0x51,0x62]
// CHECK-INST:  smt.vnpack.vv v1, v3, v5, 0x1
smt.vnpack.vv v1, v3, v5, 1 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0xab,0xd0,0x51,0x62]
// CHECK-INST:  smt.vnspack.vv	v1, v3, v5, 0x1
smt.vnspack.vv v1, v3, v5, 1 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0xab,0x90,0x51,0x42]
// CHECK-INST:  smt.vnpack4.vv	v1, v3, v5, 0x1
smt.vnpack4.vv v1, v3, v5, 1 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0xab,0xd0,0x51,0x42]
// CHECK-INST:  smt.vnspack4.vv	v1, v3, v5, 0x1
smt.vnspack4.vv v1, v3, v5, 1 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0x2b,0x11,0x52,0x66]
// CHECK-INST:  smt.vpack.vv	v2, v4, v5, 0x1
smt.vpack.vv v2, v4, v5, 1 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}

// CHECK-ENC:   encoding: [0x2b,0x51,0x52,0x66]
// CHECK-INST:  smt.vupack.vv	v2, v4, v5, 0x1
smt.vupack.vv v2, v4, v5, 1 // expected-error {{instruction requires the following: 'XSMTVDotII' (SpacemiT Vector Extension for Matrix 2.0)}}
