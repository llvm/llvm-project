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

th.vfcvt.xu.f.v v8, v4, v0.t
# CHECK-INST: th.vfcvt.xu.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x40,0x88]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 40 88 <unknown>

th.vfcvt.xu.f.v v8, v4
# CHECK-INST: th.vfcvt.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x40,0x8a]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 40 8a <unknown>

th.vfcvt.x.f.v v8, v4, v0.t
# CHECK-INST: th.vfcvt.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x40,0x88]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 94 40 88 <unknown>

th.vfcvt.x.f.v v8, v4
# CHECK-INST: th.vfcvt.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x40,0x8a]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 94 40 8a <unknown>

th.vfcvt.f.xu.v v8, v4, v0.t
# CHECK-INST: th.vfcvt.f.xu.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x41,0x88]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 41 88 <unknown>

th.vfcvt.f.xu.v v8, v4
# CHECK-INST: th.vfcvt.f.xu.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x41,0x8a]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 41 8a <unknown>

th.vfcvt.f.x.v v8, v4, v0.t
# CHECK-INST: th.vfcvt.f.x.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x41,0x88]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 94 41 88 <unknown>

th.vfcvt.f.x.v v8, v4
# CHECK-INST: th.vfcvt.f.x.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x41,0x8a]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 94 41 8a <unknown>

th.vfwcvt.xu.f.v v8, v4, v0.t
# CHECK-INST: th.vfwcvt.xu.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x44,0x88]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 44 88 <unknown>

th.vfwcvt.xu.f.v v8, v4
# CHECK-INST: th.vfwcvt.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x44,0x8a]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 44 8a <unknown>

th.vfwcvt.x.f.v v8, v4, v0.t
# CHECK-INST: th.vfwcvt.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x44,0x88]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 94 44 88 <unknown>

th.vfwcvt.x.f.v v8, v4
# CHECK-INST: th.vfwcvt.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x44,0x8a]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 94 44 8a <unknown>

th.vfwcvt.f.xu.v v8, v4, v0.t
# CHECK-INST: th.vfwcvt.f.xu.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x45,0x88]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 45 88 <unknown>

th.vfwcvt.f.xu.v v8, v4
# CHECK-INST: th.vfwcvt.f.xu.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x45,0x8a]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 45 8a <unknown>

th.vfwcvt.f.x.v v8, v4, v0.t
# CHECK-INST: th.vfwcvt.f.x.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x45,0x88]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 94 45 88 <unknown>

th.vfwcvt.f.x.v v8, v4
# CHECK-INST: th.vfwcvt.f.x.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x45,0x8a]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 94 45 8a <unknown>

th.vfwcvt.f.f.v v8, v4, v0.t
# CHECK-INST: th.vfwcvt.f.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x46,0x88]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 46 88 <unknown>

th.vfwcvt.f.f.v v8, v4
# CHECK-INST: th.vfwcvt.f.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x46,0x8a]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 46 8a <unknown>

th.vfncvt.xu.f.v v8, v4, v0.t
# CHECK-INST: th.vfncvt.xu.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x48,0x88]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 48 88 <unknown>

th.vfncvt.xu.f.v v4, v4, v0.t
# CHECK-INST: th.vfncvt.xu.f.v v4, v4, v0.t
# CHECK-ENCODING: [0x57,0x12,0x48,0x88]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 12 48 88 <unknown>

th.vfncvt.xu.f.v v8, v4
# CHECK-INST: th.vfncvt.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x48,0x8a]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 48 8a <unknown>

th.vfncvt.x.f.v v8, v4, v0.t
# CHECK-INST: th.vfncvt.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x48,0x88]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 94 48 88 <unknown>

th.vfncvt.x.f.v v8, v4
# CHECK-INST: th.vfncvt.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x48,0x8a]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 94 48 8a <unknown>

th.vfncvt.f.xu.v v8, v4, v0.t
# CHECK-INST: th.vfncvt.f.xu.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x49,0x88]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 49 88 <unknown>

th.vfncvt.f.xu.v v8, v4
# CHECK-INST: th.vfncvt.f.xu.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x49,0x8a]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 49 8a <unknown>

th.vfncvt.f.x.v v8, v4, v0.t
# CHECK-INST: th.vfncvt.f.x.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x49,0x88]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 94 49 88 <unknown>

th.vfncvt.f.x.v v8, v4
# CHECK-INST: th.vfncvt.f.x.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x49,0x8a]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 94 49 8a <unknown>

th.vfncvt.f.f.v v8, v4, v0.t
# CHECK-INST: th.vfncvt.f.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x88]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a 88 <unknown>

th.vfncvt.f.f.v v8, v4
# CHECK-INST: th.vfncvt.f.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0x8a]
# CHECK-ERROR: instruction requires the following: 'F' (Single-Precision Floating-Point), 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 14 4a 8a <unknown>
