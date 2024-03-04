# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xtheadvector %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:        | llvm-objdump -d --mattr=+xtheadvector - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvector %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

th.vmacc.vv v8, v20, v4, v0.t
# CHECK-INST: th.vmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xb4]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a b4 <unknown>

th.vmacc.vv v8, v20, v4
# CHECK-INST: th.vmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xb6]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a b6 <unknown>

th.vmacc.vx v8, a0, v4, v0.t
# CHECK-INST: th.vmacc.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xb4]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 b4 <unknown>

th.vmacc.vx v8, a0, v4
# CHECK-INST: th.vmacc.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xb6]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 b6 <unknown>

th.vnmsac.vv v8, v20, v4, v0.t
# CHECK-INST: th.vnmsac.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xbc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a bc <unknown>

th.vnmsac.vv v8, v20, v4
# CHECK-INST: th.vnmsac.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xbe]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a be <unknown>

th.vnmsac.vx v8, a0, v4, v0.t
# CHECK-INST: th.vnmsac.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 bc <unknown>

th.vnmsac.vx v8, a0, v4
# CHECK-INST: th.vnmsac.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 be <unknown>

th.vmadd.vv v8, v20, v4, v0.t
# CHECK-INST: th.vmadd.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa4]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a a4 <unknown>

th.vmadd.vv v8, v20, v4
# CHECK-INST: th.vmadd.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa6]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a a6 <unknown>

th.vmadd.vx v8, a0, v4, v0.t
# CHECK-INST: th.vmadd.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 a4 <unknown>

th.vmadd.vx v8, a0, v4
# CHECK-INST: th.vmadd.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 a6 <unknown>

th.vnmsub.vv v8, v20, v4, v0.t
# CHECK-INST: th.vnmsub.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xac]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a ac <unknown>

th.vnmsub.vv v8, v20, v4
# CHECK-INST: th.vnmsub.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xae]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a ae <unknown>

th.vnmsub.vx v8, a0, v4, v0.t
# CHECK-INST: th.vnmsub.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 ac <unknown>

th.vnmsub.vx v8, a0, v4
# CHECK-INST: th.vnmsub.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 ae <unknown>

th.vwmaccu.vv v8, v20, v4, v0.t
# CHECK-INST: th.vwmaccu.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xf0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a f0 <unknown>

th.vwmaccu.vv v8, v20, v4
# CHECK-INST: th.vwmaccu.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xf2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a f2 <unknown>

th.vwmaccu.vx v8, a0, v4, v0.t
# CHECK-INST: th.vwmaccu.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xf0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 f0 <unknown>

th.vwmaccu.vx v8, a0, v4
# CHECK-INST: th.vwmaccu.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xf2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 f2 <unknown>

th.vwmacc.vv v8, v20, v4, v0.t
# CHECK-INST: th.vwmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xf4]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a f4 <unknown>

th.vwmacc.vv v8, v20, v4
# CHECK-INST: th.vwmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xf6]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a f6 <unknown>

th.vwmacc.vx v8, a0, v4, v0.t
# CHECK-INST: th.vwmacc.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xf4]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 f4 <unknown>

th.vwmacc.vx v8, a0, v4
# CHECK-INST: th.vwmacc.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xf6]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 f6 <unknown>

th.vwmaccsu.vv v8, v20, v4, v0.t
# CHECK-INST: th.vwmaccsu.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xf8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a f8 <unknown>

th.vwmaccsu.vv v8, v20, v4
# CHECK-INST: th.vwmaccsu.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x24,0x4a,0xfa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 24 4a fa <unknown>

th.vwmaccsu.vx v8, a0, v4, v0.t
# CHECK-INST: th.vwmaccsu.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xf8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 f8 <unknown>

th.vwmaccsu.vx v8, a0, v4
# CHECK-INST: th.vwmaccsu.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xfa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 fa <unknown>

th.vwmaccus.vx v8, a0, v4, v0.t
# CHECK-INST: th.vwmaccus.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 fc <unknown>

th.vwmaccus.vx v8, a0, v4
# CHECK-INST: th.vwmaccus.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x64,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 64 45 fe <unknown>

th.vwsmaccu.vv v8, v20, v4, v0.t
# CHECK-INST: th.vwsmaccu.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xf0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a f0 <unknown>

th.vwsmaccu.vv v8, v20, v4
# CHECK-INST: th.vwsmaccu.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x04,0x4a,0xf2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a f2 <unknown>

th.vwsmaccu.vx v8, a0, v4, v0.t
# CHECK-INST: th.vwsmaccu.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xf0]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 f0 <unknown>

th.vwsmaccu.vx v8, a0, v4
# CHECK-INST: th.vwsmaccu.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x44,0x45,0xf2]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 f2 <unknown>

th.vwsmacc.vv v8, v20, v4, v0.t
# CHECK-INST: th.vwsmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xf4]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a f4 <unknown>

th.vwsmacc.vv v8, v20, v4
# CHECK-INST: th.vwsmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x04,0x4a,0xf6]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a f6 <unknown>

th.vwsmacc.vx v8, a0, v4, v0.t
# CHECK-INST: th.vwsmacc.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xf4]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 f4 <unknown>

th.vwsmacc.vx v8, a0, v4
# CHECK-INST: th.vwsmacc.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x44,0x45,0xf6]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 f6 <unknown>

th.vwsmaccsu.vv v8, v20, v4, v0.t
# CHECK-INST: th.vwsmaccsu.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xf8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a f8 <unknown>

th.vwsmaccsu.vv v8, v20, v4
# CHECK-INST: th.vwsmaccsu.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x04,0x4a,0xfa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 04 4a fa <unknown>

th.vwsmaccsu.vx v8, a0, v4, v0.t
# CHECK-INST: th.vwsmaccsu.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xf8]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 f8 <unknown>

th.vwsmaccsu.vx v8, a0, v4
# CHECK-INST: th.vwsmaccsu.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x44,0x45,0xfa]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 fa <unknown>

th.vwsmaccus.vx v8, a0, v4, v0.t
# CHECK-INST: th.vwsmaccus.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 fc <unknown>

th.vwsmaccus.vx v8, a0, v4
# CHECK-INST: th.vwsmaccus.vx v8, a0, v4
# CHECK-ENCODING: [0x57,0x44,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'xtheadvector' (T-Head Base Vector Instructions){{$}}
# CHECK-UNKNOWN: 57 44 45 fe <unknown>