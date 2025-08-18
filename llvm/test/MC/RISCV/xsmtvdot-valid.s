# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+xsmtvdot %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xsmtvdot %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+xsmtvdot %s \
# RUN:        | llvm-objdump -d  --mattr=+xsmtvdot - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xsmtvdot %s \
# RUN:        | llvm-objdump -d  --mattr=+xsmtvdot - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+xsmtvdot %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xsmtvdot %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

# CHECK-INST: smt.vmadot  v16, v0, v8
# CHECK-ENCODING: [0x2b,0x38,0x80,0xe2]
# CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
# CHECK-UNKNOWN: e280382b <unknown>
smt.vmadot   v16, v0, v8

# CHECK-INST: smt.vmadotu v18, v1, v9
# CHECK-ENCODING: [0x2b,0x89,0x90,0xe2]
# CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
# CHECK-UNKNOWN: e290892b <unknown>
smt.vmadotu  v18, v1, v9

# CHECK-INST: smt.vmadotsu        v20, v2, v10
# CHECK-ENCODING: [0x2b,0x2a,0xa1,0xe2]
# CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
# CHECK-UNKNOWN: e2a12a2b <unknown>
smt.vmadotsu v20, v2, v10

# CHECK-INST: smt.vmadotus        v22, v3, v11
# CHECK-ENCODING: [0x2b,0x9b,0xb1,0xe2]
# CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
# CHECK-UNKNOWN: e2b19b2b <unknown>
smt.vmadotus v22, v3, v11

# CHECK-INST: smt.vmadot1 v24, v16, v12
# CHECK-ENCODING: [0x2b,0x3c,0xc8,0xe6]
# CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
# CHECK-UNKNOWN: e6c83c2b <unknown>
smt.vmadot1   v24, v16, v12

# CHECK-INST: smt.vmadot1u        v26, v18, v13
# CHECK-ENCODING: [0x2b,0x0d,0xd9,0xe6]
# CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
# CHECK-UNKNOWN: e6d90d2b <unknown>
smt.vmadot1u  v26, v18, v13

# CHECK-INST: smt.vmadot1su       v28, v20, v14
# CHECK-ENCODING: [0x2b,0x2e,0xea,0xe6]
# CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
# CHECK-UNKNOWN: e6ea2e2b <unknown>
smt.vmadot1su v28, v20, v14

# CHECK-INST: smt.vmadot1us       v30, v22, v15
# CHECK-ENCODING: [0x2b,0x1f,0xfb,0xe6]
# CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
# CHECK-UNKNOWN: e6fb1f2b <unknown>
smt.vmadot1us v30, v22, v15

# CHECK-INST: smt.vmadot2 v0, v24, v4
# CHECK-ENCODING: [0x2b,0x70,0x4c,0xe6]
# CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
# CHECK-UNKNOWN: e64c702b <unknown>
smt.vmadot2   v0, v24, v4

# CHECK-INST: smt.vmadot2u        v2, v26, v5
# CHECK-ENCODING: [0x2b,0x41,0x5d,0xe6]
# CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
# CHECK-UNKNOWN: e65d412b <unknown>
smt.vmadot2u  v2, v26, v5

# CHECK-INST: smt.vmadot2su       v4, v28, v6
# CHECK-ENCODING: [0x2b,0x62,0x6e,0xe6]
# CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
# CHECK-UNKNOWN: e66e622b <unknown>
smt.vmadot2su v4, v28, v6

# CHECK-INST: smt.vmadot2us       v6, v30, v7
# CHECK-ENCODING: [0x2b,0x53,0x7f,0xe6]
# CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
# CHECK-UNKNOWN: e67f532b <unknown>
smt.vmadot2us v6, v30, v7

# CHECK-INST: smt.vmadot3 v8, v0, v8
# CHECK-ENCODING: [0x2b,0xb4,0x80,0xe6]
# CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
# CHECK-UNKNOWN: e680b42b <unknown>
smt.vmadot3   v8, v0, v8

# CHECK-INST: smt.vmadot3u        v10, v2, v9
# CHECK-ENCODING: [0x2b,0x85,0x91,0xe6]
# CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
# CHECK-UNKNOWN: e691852b <unknown>
smt.vmadot3u  v10, v2, v9

# CHECK-INST: smt.vmadot3su       v12, v4, v10
# CHECK-ENCODING: [0x2b,0xa6,0xa2,0xe6]
# CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
# CHECK-UNKNOWN: e6a2a62b <unknown>
smt.vmadot3su v12, v4, v10

# CHECK-INST: smt.vmadot3us       v14, v6, v11
# CHECK-ENCODING: [0x2b,0x97,0xb3,0xe6]
# CHECK-ERROR: instruction requires the following: 'XSMTVDot' (SpacemiT Vector Dot Product Extension){{$}}
# CHECK-UNKNOWN: e6b3972b <unknown>
smt.vmadot3us v14, v6, v11