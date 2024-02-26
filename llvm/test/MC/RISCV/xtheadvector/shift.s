# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xtheadvector %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:        | llvm-objdump -d --mattr=+xtheadvector --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

th.vsll.vv v8, v4, v20, v0.t
# CHECK-INST: th.vsll.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x94]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 94 <unknown>

th.vsll.vv v8, v4, v20
# CHECK-INST: th.vsll.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x96]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 96 <unknown>

th.vsll.vx v8, v4, a0, v0.t
# CHECK-INST: th.vsll.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x94]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 94 <unknown>

th.vsll.vx v8, v4, a0
# CHECK-INST: th.vsll.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x96]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 96 <unknown>

th.vsll.vi v8, v4, 31, v0.t
# CHECK-INST: th.vsll.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x94]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f 94 <unknown>

th.vsll.vi v8, v4, 31
# CHECK-INST: th.vsll.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x96]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f 96 <unknown>

th.vsrl.vv v8, v4, v20, v0.t
# CHECK-INST: th.vsrl.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a a0 <unknown>

th.vsrl.vv v8, v4, v20
# CHECK-INST: th.vsrl.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a a2 <unknown>

th.vsrl.vx v8, v4, a0, v0.t
# CHECK-INST: th.vsrl.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xa0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 a0 <unknown>

th.vsrl.vx v8, v4, a0
# CHECK-INST: th.vsrl.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xa2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 a2 <unknown>

th.vsrl.vi v8, v4, 31, v0.t
# CHECK-INST: th.vsrl.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f a0 <unknown>

th.vsrl.vi v8, v4, 31
# CHECK-INST: th.vsrl.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f a2 <unknown>

th.vsra.vv v8, v4, v20, v0.t
# CHECK-INST: th.vsra.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa4]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a a4 <unknown>

th.vsra.vv v8, v4, v20
# CHECK-INST: th.vsra.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa6]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a a6 <unknown>

th.vsra.vx v8, v4, a0, v0.t
# CHECK-INST: th.vsra.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 a4 <unknown>

th.vsra.vx v8, v4, a0
# CHECK-INST: th.vsra.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 a6 <unknown>

th.vsra.vi v8, v4, 31, v0.t
# CHECK-INST: th.vsra.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa4]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f a4 <unknown>

th.vsra.vi v8, v4, 31
# CHECK-INST: th.vsra.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa6]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f a6 <unknown>

th.vnsrl.vv v8, v4, v20, v0.t
# CHECK-INST: th.vnsrl.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a b0 <unknown>

th.vnsrl.vv v4, v4, v20, v0.t
# CHECK-INST: th.vnsrl.vv v4, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x02,0x4a,0xb0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 02 4a b0 <unknown>

th.vnsrl.vv v8, v4, v20
# CHECK-INST: th.vnsrl.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a b2 <unknown>

th.vnsrl.vx v8, v4, a0, v0.t
# CHECK-INST: th.vnsrl.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xb0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 b0 <unknown>

th.vnsrl.vx v8, v4, a0
# CHECK-INST: th.vnsrl.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xb2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 b2 <unknown>

th.vnsrl.vi v8, v4, 31, v0.t
# CHECK-INST: th.vnsrl.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f b0 <unknown>

th.vnsrl.vi v8, v4, 31
# CHECK-INST: th.vnsrl.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f b2 <unknown>

th.vnsra.vv v8, v4, v20, v0.t
# CHECK-INST: th.vnsra.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb4]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a b4 <unknown>

th.vnsra.vv v8, v4, v20
# CHECK-INST: th.vnsra.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb6]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a b6 <unknown>

th.vnsra.vx v8, v4, a0, v0.t
# CHECK-INST: th.vnsra.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xb4]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 b4 <unknown>

th.vnsra.vx v8, v4, a0
# CHECK-INST: th.vnsra.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xb6]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 b6 <unknown>

th.vnsra.vi v8, v4, 31, v0.t
# CHECK-INST: th.vnsra.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb4]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f b4 <unknown>

th.vnsra.vi v8, v4, 31
# CHECK-INST: th.vnsra.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb6]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f b6 <unknown>

th.vssrl.vv v8, v4, v20, v0.t
# CHECK-INST: th.vssrl.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a a8 <unknown>

th.vssrl.vv v8, v4, v20
# CHECK-INST: th.vssrl.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xaa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a aa <unknown>

th.vssrl.vx v8, v4, a0, v0.t
# CHECK-INST: th.vssrl.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xa8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 a8 <unknown>

th.vssrl.vx v8, v4, a0
# CHECK-INST: th.vssrl.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xaa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 aa <unknown>

th.vssrl.vi v8, v4, 31, v0.t
# CHECK-INST: th.vssrl.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f a8 <unknown>

th.vssrl.vi v8, v4, 31
# CHECK-INST: th.vssrl.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xaa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f aa <unknown>

th.vssra.vv v8, v4, v20, v0.t
# CHECK-INST: th.vssra.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xac]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a ac <unknown>

th.vssra.vv v8, v4, v20
# CHECK-INST: th.vssra.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xae]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a ae <unknown>

th.vssra.vx v8, v4, a0, v0.t
# CHECK-INST: th.vssra.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 ac <unknown>

th.vssra.vx v8, v4, a0
# CHECK-INST: th.vssra.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 ae <unknown>

th.vssra.vi v8, v4, 31, v0.t
# CHECK-INST: th.vssra.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xac]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f ac <unknown>

th.vssra.vi v8, v4, 31
# CHECK-INST: th.vssra.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xae]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 4f ae <unknown>
