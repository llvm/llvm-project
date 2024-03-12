# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xtheadvector %s \
# RUN:         --mattr=+f \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:         --mattr=+f \
# RUN:        | llvm-objdump -d --mattr=+xtheadvector --mattr=+f - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:         --mattr=+f \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

th.vfmacc.vv v8, v20, v4, v0.t
# CHECK-INST: th.vfmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xb0]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a b0 <unknown>

th.vfmacc.vv v8, v20, v4
# CHECK-INST: th.vfmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xb2]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a b2 <unknown>

th.vfmacc.vf v8, fa0, v4, v0.t
# CHECK-INST: th.vfmacc.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xb0]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 b0 <unknown>

th.vfmacc.vf v8, fa0, v4
# CHECK-INST: th.vfmacc.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xb2]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 b2 <unknown>

th.vfnmacc.vv v8, v20, v4, v0.t
# CHECK-INST: th.vfnmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xb4]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a b4 <unknown>

th.vfnmacc.vv v8, v20, v4
# CHECK-INST: th.vfnmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xb6]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a b6 <unknown>

th.vfnmacc.vf v8, fa0, v4, v0.t
# CHECK-INST: th.vfnmacc.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xb4]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 b4 <unknown>

th.vfnmacc.vf v8, fa0, v4
# CHECK-INST: th.vfnmacc.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xb6]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 b6 <unknown>

th.vfmsac.vv v8, v20, v4, v0.t
# CHECK-INST: th.vfmsac.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xb8]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a b8 <unknown>

th.vfmsac.vv v8, v20, v4
# CHECK-INST: th.vfmsac.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xba]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a ba <unknown>

th.vfmsac.vf v8, fa0, v4, v0.t
# CHECK-INST: th.vfmsac.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xb8]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 b8 <unknown>

th.vfmsac.vf v8, fa0, v4
# CHECK-INST: th.vfmsac.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xba]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 ba <unknown>

th.vfnmsac.vv v8, v20, v4, v0.t
# CHECK-INST: th.vfnmsac.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xbc]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a bc <unknown>

th.vfnmsac.vv v8, v20, v4
# CHECK-INST: th.vfnmsac.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xbe]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a be <unknown>

th.vfnmsac.vf v8, fa0, v4, v0.t
# CHECK-INST: th.vfnmsac.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 bc <unknown>

th.vfnmsac.vf v8, fa0, v4
# CHECK-INST: th.vfnmsac.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 be <unknown>

th.vfmadd.vv v8, v20, v4, v0.t
# CHECK-INST: th.vfmadd.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xa0]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a a0 <unknown>

th.vfmadd.vv v8, v20, v4
# CHECK-INST: th.vfmadd.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xa2]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a a2 <unknown>

th.vfmadd.vf v8, fa0, v4, v0.t
# CHECK-INST: th.vfmadd.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xa0]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 a0 <unknown>

th.vfmadd.vf v8, fa0, v4
# CHECK-INST: th.vfmadd.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xa2]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 a2 <unknown>

th.vfnmadd.vv v8, v20, v4, v0.t
# CHECK-INST: th.vfnmadd.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xa4]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a a4 <unknown>

th.vfnmadd.vv v8, v20, v4
# CHECK-INST: th.vfnmadd.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xa6]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a a6 <unknown>

th.vfnmadd.vf v8, fa0, v4, v0.t
# CHECK-INST: th.vfnmadd.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 a4 <unknown>

th.vfnmadd.vf v8, fa0, v4
# CHECK-INST: th.vfnmadd.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 a6 <unknown>

th.vfmsub.vv v8, v20, v4, v0.t
# CHECK-INST: th.vfmsub.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xa8]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a a8 <unknown>

th.vfmsub.vv v8, v20, v4
# CHECK-INST: th.vfmsub.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xaa]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a aa <unknown>

th.vfmsub.vf v8, fa0, v4, v0.t
# CHECK-INST: th.vfmsub.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xa8]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 a8 <unknown>

th.vfmsub.vf v8, fa0, v4
# CHECK-INST: th.vfmsub.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xaa]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 aa <unknown>

th.vfnmsub.vv v8, v20, v4, v0.t
# CHECK-INST: th.vfnmsub.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xac]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a ac <unknown>

th.vfnmsub.vv v8, v20, v4
# CHECK-INST: th.vfnmsub.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xae]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a ae <unknown>

th.vfnmsub.vf v8, fa0, v4, v0.t
# CHECK-INST: th.vfnmsub.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 ac <unknown>

th.vfnmsub.vf v8, fa0, v4
# CHECK-INST: th.vfnmsub.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 ae <unknown>

th.vfwmacc.vv v8, v20, v4, v0.t
# CHECK-INST: th.vfwmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xf0]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a f0 <unknown>

th.vfwmacc.vv v8, v20, v4
# CHECK-INST: th.vfwmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xf2]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a f2 <unknown>

th.vfwmacc.vf v8, fa0, v4, v0.t
# CHECK-INST: th.vfwmacc.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xf0]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 f0 <unknown>

th.vfwmacc.vf v8, fa0, v4
# CHECK-INST: th.vfwmacc.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xf2]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 f2 <unknown>

th.vfwnmacc.vv v8, v20, v4, v0.t
# CHECK-INST: th.vfwnmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xf4]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a f4 <unknown>

th.vfwnmacc.vv v8, v20, v4
# CHECK-INST: th.vfwnmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xf6]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a f6 <unknown>

th.vfwnmacc.vf v8, fa0, v4, v0.t
# CHECK-INST: th.vfwnmacc.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xf4]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 f4 <unknown>

th.vfwnmacc.vf v8, fa0, v4
# CHECK-INST: th.vfwnmacc.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xf6]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 f6 <unknown>

th.vfwmsac.vv v8, v20, v4, v0.t
# CHECK-INST: th.vfwmsac.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xf8]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a f8 <unknown>

th.vfwmsac.vv v8, v20, v4
# CHECK-INST: th.vfwmsac.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xfa]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a fa <unknown>

th.vfwmsac.vf v8, fa0, v4, v0.t
# CHECK-INST: th.vfwmsac.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xf8]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 f8 <unknown>

th.vfwmsac.vf v8, fa0, v4
# CHECK-INST: th.vfwmsac.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xfa]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 fa <unknown>

th.vfwnmsac.vv v8, v20, v4, v0.t
# CHECK-INST: th.vfwnmsac.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xfc]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a fc <unknown>

th.vfwnmsac.vv v8, v20, v4
# CHECK-INST: th.vfwnmsac.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xfe]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a fe <unknown>

th.vfwnmsac.vf v8, fa0, v4, v0.t
# CHECK-INST: th.vfwnmsac.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 fc <unknown>

th.vfwnmsac.vf v8, fa0, v4
# CHECK-INST: th.vfwnmsac.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 54 45 fe <unknown>
