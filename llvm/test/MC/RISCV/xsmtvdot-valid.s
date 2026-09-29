# RUN: llvm-mc -triple=riscv64 -show-encoding -mcpu=spacemit-x60 %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENC

# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

# RUN: llvm-mc -triple=riscv64 -filetype=obj -mcpu=spacemit-x60 %s \
# RUN:        | llvm-objdump -d  --mcpu=spacemit-x60 - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST



// CHECK-ENC:   encoding: [0x2b,0x38,0x80,0xe2]
// CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
// CHECK-INST:  smt.vmadot  v16, v0, v8
smt.vmadot   v16, v0, v8

// CHECK-ENC:   encoding: [0x2b,0x89,0x90,0xe2]
// CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
// CHECK-INST:  smt.vmadotu	v18, v1, v9 
smt.vmadotu  v18, v1, v9

// CHECK-ENC:   encoding: [0x2b,0x2a,0xa1,0xe2]
// CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
// CHECK-INST:  smt.vmadotsu	v20, v2, v10
smt.vmadotsu v20, v2, v10

// CHECK-ENC:   encoding: [0x2b,0x9b,0xb1,0xe2]
// CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
// CHECK-INST:  smt.vmadotus	v22, v3, v11
smt.vmadotus v22, v3, v11


// CHECK-ENC:   encoding: [0x2b,0x3c,0xc8,0xe6]
// CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
// CHECK-INST:  smt.vmadot1	v24, v16, v12
smt.vmadot1   v24, v16, v12


// CHECK-ENC:   encoding: [0x2b,0x0d,0xd9,0xe6]
// CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
// CHECK-INST:  smt.vmadot1u	v26, v18, v13
smt.vmadot1u  v26, v18, v13

// CHECK-ENC:   encoding: [0x2b,0x2e,0xea,0xe6]
// CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
// CHECK-INST:  smt.vmadot1su	v28, v20, v14
smt.vmadot1su v28, v20, v14

// CHECK-ENC:   encoding: [0x2b,0x1f,0xfb,0xe6]
// CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
// CHECK-INST:  smt.vmadot1us	v30, v22, v15
smt.vmadot1us v30, v22, v15

// CHECK-ENC:   encoding: [0x2b,0x70,0x4c,0xe6]
// CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
// CHECK-INST:  smt.vmadot2	v0, v24, v4
smt.vmadot2   v0, v24, v4

// CHECK-ENC:   encoding: [0x2b,0x41,0x5d,0xe6]
// CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
// CHECK-INST:  smt.vmadot2u	v2, v26, v5
smt.vmadot2u  v2, v26, v5

// CHECK-ENC:   encoding: [0x2b,0x62,0x6e,0xe6]
// CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
// CHECK-INST:  smt.vmadot2su	v4, v28, v6
smt.vmadot2su v4, v28, v6

// CHECK-ENC:   encoding: [0x2b,0x53,0x8f,0xe6]
// CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
// CHECK-INST:  smt.vmadot2us	v6, v30, v8
smt.vmadot2us v6, v30, v8

// CHECK-ENC:   encoding: [0x2b,0xb1,0x80,0xe6]
// CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
// CHECK-INST:  smt.vmadot3	v2, v0, v8 
smt.vmadot3   v2, v0, v8

// CHECK-ENC:   encoding: [0x2b,0x85,0x91,0xe6]
// CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
// CHECK-INST:  smt.vmadot3u	v10, v2, v9
smt.vmadot3u  v10, v2, v9


// CHECK-ENC:   encoding: [0x2b,0xa6,0xa2,0xe6]
// CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
// CHECK-INST:  smt.vmadot3su	v12, v4, v10
smt.vmadot3su v12, v4, v10

// CHECK-ENC:   encoding: [0x2b,0x97,0xb3,0xe6]
// CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
// CHECK-INST: smt.vmadot3us	v14, v6, v11
smt.vmadot3us v14, v6, v11
