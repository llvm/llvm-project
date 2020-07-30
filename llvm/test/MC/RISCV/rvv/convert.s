# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vfcvt.xu.f.v v8, v4, v0.t
# CHECK-INST: vfcvt.xu.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x40,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 40 88 <unknown>

vfcvt.xu.f.v v8, v4
# CHECK-INST: vfcvt.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x40,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 40 8a <unknown>

vfcvt.x.f.v v8, v4, v0.t
# CHECK-INST: vfcvt.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x40,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 94 40 88 <unknown>

vfcvt.x.f.v v8, v4
# CHECK-INST: vfcvt.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x40,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 94 40 8a <unknown>

vfcvt.f.xu.v v8, v4, v0.t
# CHECK-INST: vfcvt.f.xu.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x41,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 41 88 <unknown>

vfcvt.f.xu.v v8, v4
# CHECK-INST: vfcvt.f.xu.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x41,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 41 8a <unknown>

vfcvt.f.x.v v8, v4, v0.t
# CHECK-INST: vfcvt.f.x.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x41,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 94 41 88 <unknown>

vfcvt.f.x.v v8, v4
# CHECK-INST: vfcvt.f.x.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x41,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 94 41 8a <unknown>

vfwcvt.xu.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.xu.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x44,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 44 88 <unknown>

vfwcvt.xu.f.v v8, v4
# CHECK-INST: vfwcvt.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x44,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 44 8a <unknown>

vfwcvt.x.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x44,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 94 44 88 <unknown>

vfwcvt.x.f.v v8, v4
# CHECK-INST: vfwcvt.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x44,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 94 44 8a <unknown>

vfwcvt.f.xu.v v8, v4, v0.t
# CHECK-INST: vfwcvt.f.xu.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x45,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 45 88 <unknown>

vfwcvt.f.xu.v v8, v4
# CHECK-INST: vfwcvt.f.xu.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x45,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 45 8a <unknown>

vfwcvt.f.x.v v8, v4, v0.t
# CHECK-INST: vfwcvt.f.x.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x45,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 94 45 88 <unknown>

vfwcvt.f.x.v v8, v4
# CHECK-INST: vfwcvt.f.x.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x45,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 94 45 8a <unknown>

vfwcvt.f.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.f.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x46,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 46 88 <unknown>

vfwcvt.f.f.v v8, v4
# CHECK-INST: vfwcvt.f.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x46,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 46 8a <unknown>

vfncvt.xu.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.xu.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x48,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 48 88 <unknown>

vfncvt.xu.f.w v8, v4
# CHECK-INST: vfncvt.xu.f.w v8, v4
# CHECK-ENCODING: [0x57,0x14,0x48,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 48 8a <unknown>

vfncvt.x.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.x.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x48,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 94 48 88 <unknown>

vfncvt.x.f.w v8, v4
# CHECK-INST: vfncvt.x.f.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x48,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 94 48 8a <unknown>

vfncvt.f.xu.w v8, v4, v0.t
# CHECK-INST: vfncvt.f.xu.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x49,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 49 88 <unknown>

vfncvt.f.xu.w v8, v4
# CHECK-INST: vfncvt.f.xu.w v8, v4
# CHECK-ENCODING: [0x57,0x14,0x49,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 49 8a <unknown>

vfncvt.f.x.w v8, v4, v0.t
# CHECK-INST: vfncvt.f.x.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x49,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 94 49 88 <unknown>

vfncvt.f.x.w v8, v4
# CHECK-INST: vfncvt.f.x.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x49,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 94 49 8a <unknown>

vfncvt.f.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.f.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 4a 88 <unknown>

vfncvt.f.f.w v8, v4
# CHECK-INST: vfncvt.f.f.w v8, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 14 4a 8a <unknown>

vfncvt.rod.f.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.rod.f.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4a,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 94 4a 88 <unknown>

vfncvt.rod.f.f.w v8, v4
# CHECK-INST: vfncvt.rod.f.f.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x4a,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Instructions)
# CHECK-UNKNOWN: 57 94 4a 8a <unknown>
