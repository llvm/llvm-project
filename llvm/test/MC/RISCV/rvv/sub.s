# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vsub.vv v8, v4, v20, v0.t
# CHECK-INST: vsub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a 08 <unknown>

vsub.vv v8, v4, v20
# CHECK-INST: vsub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a 0a <unknown>

vsub.vx v8, v4, a0, v0.t
# CHECK-INST: vsub.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 08 <unknown>

vsub.vx v8, v4, a0
# CHECK-INST: vsub.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 0a <unknown>

vrsub.vx v8, v4, a0, v0.t
# CHECK-INST: vrsub.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 0c <unknown>

vrsub.vx v8, v4, a0
# CHECK-INST: vrsub.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 0e <unknown>

vrsub.vi v8, v4, 15, v0.t
# CHECK-INST: vrsub.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 b4 47 0c <unknown>

vrsub.vi v8, v4, 15
# CHECK-INST: vrsub.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 b4 47 0e <unknown>

vwsubu.vv v8, v4, v20, v0.t
# CHECK-INST: vwsubu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xc8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a c8 <unknown>

vwsubu.vv v8, v4, v20
# CHECK-INST: vwsubu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xca]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a ca <unknown>

vwsubu.vx v8, v4, a0, v0.t
# CHECK-INST: vwsubu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xc8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 64 45 c8 <unknown>

vwsubu.vx v8, v4, a0
# CHECK-INST: vwsubu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xca]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 64 45 ca <unknown>

vwsub.vv v8, v4, v20, v0.t
# CHECK-INST: vwsub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xcc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a cc <unknown>

vwsub.vv v8, v4, v20
# CHECK-INST: vwsub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xce]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a ce <unknown>

vwsub.vx v8, v4, a0, v0.t
# CHECK-INST: vwsub.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 64 45 cc <unknown>

vwsub.vx v8, v4, a0
# CHECK-INST: vwsub.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 64 45 ce <unknown>

vwsubu.wv v8, v4, v20, v0.t
# CHECK-INST: vwsubu.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xd8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a d8 <unknown>

vwsubu.wv v8, v4, v20
# CHECK-INST: vwsubu.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xda]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a da <unknown>

vwsubu.wx v8, v4, a0, v0.t
# CHECK-INST: vwsubu.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xd8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 64 45 d8 <unknown>

vwsubu.wx v8, v4, a0
# CHECK-INST: vwsubu.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xda]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 64 45 da <unknown>

vwsub.wv v8, v4, v20, v0.t
# CHECK-INST: vwsub.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xdc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a dc <unknown>

vwsub.wv v8, v4, v20
# CHECK-INST: vwsub.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xde]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a de <unknown>

vwsub.wx v8, v4, a0, v0.t
# CHECK-INST: vwsub.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xdc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 64 45 dc <unknown>

vwsub.wx v8, v4, a0
# CHECK-INST: vwsub.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xde]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 64 45 de <unknown>

vsbc.vvm v8, v4, v20, v0
# CHECK-INST: vsbc.vvm v8, v4, v20, v0
# CHECK-ENCODING: [0x57,0x04,0x4a,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a 48 <unknown>

vsbc.vxm v8, v4, a0, v0
# CHECK-INST: vsbc.vxm v8, v4, a0, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 48 <unknown>

vmsbc.vvm v8, v4, v20, v0
# CHECK-INST: vmsbc.vvm v8, v4, v20, v0
# CHECK-ENCODING: [0x57,0x04,0x4a,0x4c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a 4c <unknown>

vmsbc.vxm v8, v4, a0, v0
# CHECK-INST: vmsbc.vxm v8, v4, a0, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 4c <unknown>

vmsbc.vv v8, v4, v20
# CHECK-INST: vmsbc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x4e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a 4e <unknown>

vmsbc.vx v8, v4, a0
# CHECK-INST: vmsbc.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 4e <unknown>

vssubu.vv v8, v4, v20, v0.t
# CHECK-INST: vssubu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a 88 <unknown>

vssubu.vv v8, v4, v20
# CHECK-INST: vssubu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a 8a <unknown>

vssubu.vx v8, v4, a0, v0.t
# CHECK-INST: vssubu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 88 <unknown>

vssubu.vx v8, v4, a0
# CHECK-INST: vssubu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 8a <unknown>

vssub.vv v8, v4, v20, v0.t
# CHECK-INST: vssub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x8c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a 8c <unknown>

vssub.vv v8, v4, v20
# CHECK-INST: vssub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x8e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a 8e <unknown>

vssub.vx v8, v4, a0, v0.t
# CHECK-INST: vssub.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 8c <unknown>

vssub.vx v8, v4, a0
# CHECK-INST: vssub.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 8e <unknown>

vasub.vv v8, v4, v20, v0.t
# CHECK-INST: vasub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x2c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 2c <unknown>

vasub.vv v8, v4, v20
# CHECK-INST: vasub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x2e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 2e <unknown>

vasub.vx v8, v4, a0, v0.t
# CHECK-INST: vasub.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 64 45 2c <unknown>

vasub.vx v8, v4, a0
# CHECK-INST: vasub.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 64 45 2e <unknown>

vasubu.vv v8, v4, v20, v0.t
# CHECK-INST: vasubu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x28]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 28 <unknown>

vasubu.vv v8, v4, v20
# CHECK-INST: vasubu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x2a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 24 4a 2a <unknown>

vasubu.vx v8, v4, a0, v0.t
# CHECK-INST: vasubu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x28]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 64 45 28 <unknown>

vasubu.vx v8, v4, a0
# CHECK-INST: vasubu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x2a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 64 45 2a <unknown>
