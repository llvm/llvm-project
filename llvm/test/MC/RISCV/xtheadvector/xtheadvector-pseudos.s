# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+f,+a,+xtheadvector,+xtheadzvamo %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

th.vmmv.m v4, v8
# CHECK-INST: th.vmcpy.m v4, v8
# CHECK-ENCODING: [0x57,0x22,0x84,0x66]

th.vneg.v v4, v8
# CHECK-INST: th.vneg.v v4, v8
# CHECK-ENCODING: [0x57,0x42,0x80,0x0e]

th.vncvt.x.x.v v4, v8
# CHECK-INST: th.vnsrl.vx v4, v8, zero
# CHECK-ENCODING: [0x57,0x42,0x80,0xb2]

th.vncvt.x.x.v v4, v8, v0.t
# CHECK-INST: th.vncvt.x.x.v v4, v8, v0.t
# CHECK-ENCODING: [0x57,0x42,0x80,0xb0]

th.vfneg.v v4, v8
# CHECK-INST: th.vfneg.v v4, v8
# CHECK-ENCODING: [0x57,0x12,0x84,0x26]

th.vfabs.v v4, v8
# CHECK-INST: th.vfabs.v v4, v8
# CHECK-ENCODING: [0x57,0x12,0x84,0x2a]
