# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xtheadvector %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:        | llvm-objdump -d --mattr=+xtheadvector --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

th.vsub.vv v8, v4, v20, v0.t
# CHECK-INST: th.vsub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x08]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 08 <unknown>

th.vsub.vv v8, v4, v20
# CHECK-INST: th.vsub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x0a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 0a <unknown>

th.vsub.vx v8, v4, a0, v0.t
# CHECK-INST: th.vsub.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x08]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 08 <unknown>

th.vsub.vx v8, v4, a0
# CHECK-INST: th.vsub.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x0a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 0a <unknown>

th.vrsub.vx v8, v4, a0, v0.t
# CHECK-INST: th.vrsub.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 0c <unknown>

th.vrsub.vx v8, v4, a0
# CHECK-INST: th.vrsub.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 0e <unknown>

th.vrsub.vi v8, v4, 15, v0.t
# CHECK-INST: th.vrsub.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x0c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 0c <unknown>

th.vrsub.vi v8, v4, 15
# CHECK-INST: th.vrsub.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x0e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 b4 47 0e <unknown>

th.vwsubu.vv v8, v4, v20, v0.t
# CHECK-INST: th.vwsubu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xc8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a c8 <unknown>

th.vwsubu.vv v8, v4, v20
# CHECK-INST: th.vwsubu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xca]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a ca <unknown>

th.vwsubu.vx v8, v4, a0, v0.t
# CHECK-INST: th.vwsubu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xc8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 c8 <unknown>

th.vwsubu.vx v8, v4, a0
# CHECK-INST: th.vwsubu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xca]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 ca <unknown>

th.vwsub.vv v8, v4, v20, v0.t
# CHECK-INST: th.vwsub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xcc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a cc <unknown>

th.vwsub.vv v8, v4, v20
# CHECK-INST: th.vwsub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xce]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a ce <unknown>

th.vwsub.vx v8, v4, a0, v0.t
# CHECK-INST: th.vwsub.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 cc <unknown>

th.vwsub.vx v8, v4, a0
# CHECK-INST: th.vwsub.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 ce <unknown>

th.vwsubu.wv v8, v4, v20, v0.t
# CHECK-INST: th.vwsubu.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xd8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a d8 <unknown>

th.vwsubu.wv v8, v4, v20
# CHECK-INST: th.vwsubu.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xda]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a da <unknown>

th.vwsubu.wx v8, v4, a0, v0.t
# CHECK-INST: th.vwsubu.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xd8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 d8 <unknown>

th.vwsubu.wx v8, v4, a0
# CHECK-INST: th.vwsubu.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xda]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 da <unknown>

th.vwsub.wv v8, v4, v20, v0.t
# CHECK-INST: th.vwsub.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xdc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a dc <unknown>

th.vwsub.wv v8, v4, v20
# CHECK-INST: th.vwsub.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xde]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a de <unknown>

th.vwsub.wx v8, v4, a0, v0.t
# CHECK-INST: th.vwsub.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xdc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 dc <unknown>

th.vwsub.wx v8, v4, a0
# CHECK-INST: th.vwsub.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xde]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 de <unknown>

th.vsbc.vvm v8, v4, v20, v0
# CHECK-INST: th.vsbc.vvm v8, v4, v20, v0
# CHECK-ENCODING: [0x57,0x04,0x4a,0x4a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 4a <unknown>

th.vsbc.vvm v4, v4, v20, v0
# CHECK-INST: th.vsbc.vvm v4, v4, v20, v0
# CHECK-ENCODING: [0x57,0x02,0x4a,0x4a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 02 4a 4a <unknown>

th.vsbc.vvm v8, v4, v8, v0
# CHECK-INST: th.vsbc.vvm v8, v4, v8, v0
# CHECK-ENCODING: [0x57,0x04,0x44,0x4a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 44 4a <unknown>

th.vsbc.vxm v8, v4, a0, v0
# CHECK-INST: th.vsbc.vxm v8, v4, a0, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x4a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 4a <unknown>

th.vmsbc.vvm v8, v4, v20, v0
# CHECK-INST: th.vmsbc.vvm v8, v4, v20, v0
# CHECK-ENCODING: [0x57,0x04,0x4a,0x4e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 4e <unknown>

th.vmsbc.vvm v4, v4, v20, v0
# CHECK-INST: th.vmsbc.vvm v4, v4, v20, v0
# CHECK-ENCODING: [0x57,0x02,0x4a,0x4e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 02 4a 4e <unknown>

th.vmsbc.vvm v8, v4, v8, v0
# CHECK-INST: th.vmsbc.vvm v8, v4, v8, v0
# CHECK-ENCODING: [0x57,0x04,0x44,0x4e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 44 4e <unknown>

th.vmsbc.vxm v8, v4, a0, v0
# CHECK-INST: th.vmsbc.vxm v8, v4, a0, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 4e <unknown>

th.vssubu.vv v8, v4, v20, v0.t
# CHECK-INST: th.vssubu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x88]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 88 <unknown>

th.vssubu.vv v8, v4, v20
# CHECK-INST: th.vssubu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x8a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 8a <unknown>

th.vssubu.vx v8, v4, a0, v0.t
# CHECK-INST: th.vssubu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x88]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 88 <unknown>

th.vssubu.vx v8, v4, a0
# CHECK-INST: th.vssubu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x8a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 8a <unknown>

th.vssub.vv v8, v4, v20, v0.t
# CHECK-INST: th.vssub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x8c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 8c <unknown>

th.vssub.vv v8, v4, v20
# CHECK-INST: th.vssub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x8e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 8e <unknown>

th.vssub.vx v8, v4, a0, v0.t
# CHECK-INST: th.vssub.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 8c <unknown>

th.vssub.vx v8, v4, a0
# CHECK-INST: th.vssub.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 8e <unknown>

th.vasub.vv v8, v4, v20, v0.t
# CHECK-INST: th.vasub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x98]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 98 <unknown>

th.vasub.vv v8, v4, v20
# CHECK-INST: th.vasub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x9a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a 9a <unknown>

th.vasub.vx v8, v4, a0, v0.t
# CHECK-INST: th.vasub.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x98]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 98 <unknown>

th.vasub.vx v8, v4, a0
# CHECK-INST: th.vasub.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x9a]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 9a <unknown>
