# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vsll.vv v8, v4, v20, v0.t
# CHECK-INST: vsll.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x94]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a 94 <unknown>

vsll.vv v8, v4, v20
# CHECK-INST: vsll.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x96]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a 96 <unknown>

vsll.vx v8, v4, a0, v0.t
# CHECK-INST: vsll.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x94]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 94 <unknown>

vsll.vx v8, v4, a0
# CHECK-INST: vsll.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x96]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 96 <unknown>

vsll.vi v8, v4, 31, v0.t
# CHECK-INST: vsll.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x94]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 b4 4f 94 <unknown>

vsll.vi v8, v4, 31
# CHECK-INST: vsll.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x96]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 b4 4f 96 <unknown>

vsrl.vv v8, v4, v20, v0.t
# CHECK-INST: vsrl.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a a0 <unknown>

vsrl.vv v8, v4, v20
# CHECK-INST: vsrl.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a a2 <unknown>

vsrl.vx v8, v4, a0, v0.t
# CHECK-INST: vsrl.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xa0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 a0 <unknown>

vsrl.vx v8, v4, a0
# CHECK-INST: vsrl.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xa2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 a2 <unknown>

vsrl.vi v8, v4, 31, v0.t
# CHECK-INST: vsrl.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 b4 4f a0 <unknown>

vsrl.vi v8, v4, 31
# CHECK-INST: vsrl.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 b4 4f a2 <unknown>

vsra.vv v8, v4, v20, v0.t
# CHECK-INST: vsra.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a a4 <unknown>

vsra.vv v8, v4, v20
# CHECK-INST: vsra.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a a6 <unknown>

vsra.vx v8, v4, a0, v0.t
# CHECK-INST: vsra.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 a4 <unknown>

vsra.vx v8, v4, a0
# CHECK-INST: vsra.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 a6 <unknown>

vsra.vi v8, v4, 31, v0.t
# CHECK-INST: vsra.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 b4 4f a4 <unknown>

vsra.vi v8, v4, 31
# CHECK-INST: vsra.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 b4 4f a6 <unknown>

vnsrl.wv v8, v4, v20, v0.t
# CHECK-INST: vnsrl.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a b0 <unknown>

vnsrl.wv v8, v4, v20
# CHECK-INST: vnsrl.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a b2 <unknown>

vnsrl.wx v8, v4, a0, v0.t
# CHECK-INST: vnsrl.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xb0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 b0 <unknown>

vnsrl.wx v8, v4, a0
# CHECK-INST: vnsrl.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xb2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 b2 <unknown>

vnsrl.wi v8, v4, 31, v0.t
# CHECK-INST: vnsrl.wi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 b4 4f b0 <unknown>

vnsrl.wi v8, v4, 31
# CHECK-INST: vnsrl.wi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 b4 4f b2 <unknown>

vnsra.wv v8, v4, v20, v0.t
# CHECK-INST: vnsra.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a b4 <unknown>

vnsra.wv v8, v4, v20
# CHECK-INST: vnsra.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a b6 <unknown>

vnsra.wx v8, v4, a0, v0.t
# CHECK-INST: vnsra.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xb4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 b4 <unknown>

vnsra.wx v8, v4, a0
# CHECK-INST: vnsra.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xb6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 b6 <unknown>

vnsra.wi v8, v4, 31, v0.t
# CHECK-INST: vnsra.wi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 b4 4f b4 <unknown>

vnsra.wi v8, v4, 31
# CHECK-INST: vnsra.wi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 b4 4f b6 <unknown>

vssrl.vv v8, v4, v20, v0.t
# CHECK-INST: vssrl.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xa8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a a8 <unknown>

vssrl.vv v8, v4, v20
# CHECK-INST: vssrl.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xaa]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a aa <unknown>

vssrl.vx v8, v4, a0, v0.t
# CHECK-INST: vssrl.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xa8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 a8 <unknown>

vssrl.vx v8, v4, a0
# CHECK-INST: vssrl.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xaa]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 aa <unknown>

vssrl.vi v8, v4, 31, v0.t
# CHECK-INST: vssrl.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xa8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 b4 4f a8 <unknown>

vssrl.vi v8, v4, 31
# CHECK-INST: vssrl.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xaa]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 b4 4f aa <unknown>

vssra.vv v8, v4, v20, v0.t
# CHECK-INST: vssra.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xac]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a ac <unknown>

vssra.vv v8, v4, v20
# CHECK-INST: vssra.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xae]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 04 4a ae <unknown>

vssra.vx v8, v4, a0, v0.t
# CHECK-INST: vssra.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 ac <unknown>

vssra.vx v8, v4, a0
# CHECK-INST: vssra.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 44 45 ae <unknown>

vssra.vi v8, v4, 31, v0.t
# CHECK-INST: vssra.vi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xac]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 b4 4f ac <unknown>

vssra.vi v8, v4, 31
# CHECK-INST: vssra.vi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xae]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 b4 4f ae <unknown>
