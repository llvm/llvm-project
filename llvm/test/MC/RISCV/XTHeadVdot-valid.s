# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xtheadvdot %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvdot %s \
# RUN:        | llvm-objdump -d --mattr=+xtheadvdot - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

th.vmaqau.vv v8, v20, v4, v0.t # expected-error {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}
# CHECK-INST: th.vmaqau.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x0b,0x64,0x4a,0x88]

th.vmaqau.vv v8, v20, v4 # expected-error {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}
# CHECK-INST: th.vmaqau.vv v8, v20, v4
# CHECK-ENCODING: [0x0b,0x64,0x4a,0x8a]

th.vmaqau.vx v8, a0, v4, v0.t # expected-error {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}
# CHECK-INST: th.vmaqau.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x0b,0x64,0x45,0x8c]

th.vmaqau.vx v8, a0, v4 # expected-error {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}
# CHECK-INST: th.vmaqau.vx v8, a0, v4
# CHECK-ENCODING: [0x0b,0x64,0x45,0x8e]

th.vmaqa.vv v8, v20, v4, v0.t # expected-error {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}
# CHECK-INST: th.vmaqa.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x0b,0x64,0x4a,0x80]

th.vmaqa.vv v8, v20, v4 # expected-error {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}
# CHECK-INST: th.vmaqa.vv v8, v20, v4
# CHECK-ENCODING: [0x0b,0x64,0x4a,0x82]

th.vmaqa.vx v8, a0, v4, v0.t # expected-error {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}
# CHECK-INST: th.vmaqa.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x0b,0x64,0x45,0x84]

th.vmaqa.vx v8, a0, v4 # expected-error {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}
# CHECK-INST: th.vmaqa.vx v8, a0, v4
# CHECK-ENCODING: [0x0b,0x64,0x45,0x86]

th.vmaqasu.vv v8, v20, v4, v0.t # expected-error {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}
# CHECK-INST: th.vmaqasu.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x0b,0x64,0x4a,0x90]

th.vmaqasu.vv v8, v20, v4 # expected-error {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}
# CHECK-INST: th.vmaqasu.vv v8, v20, v4
# CHECK-ENCODING: [0x0b,0x64,0x4a,0x92]

th.vmaqasu.vx v8, a0, v4, v0.t # expected-error {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}
# CHECK-INST: th.vmaqasu.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x0b,0x64,0x45,0x94]

th.vmaqasu.vx v8, a0, v4 # expected-error {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}
# CHECK-INST: th.vmaqasu.vx v8, a0, v4
# CHECK-ENCODING: [0x0b,0x64,0x45,0x96]

th.vmaqaus.vx v8, a0, v4, v0.t # expected-error {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}
# CHECK-INST: th.vmaqaus.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x0b,0x64,0x45,0x9c]

th.vmaqaus.vx v8, a0, v4 # expected-error {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}
# CHECK-INST: th.vmaqaus.vx v8, a0, v4
# CHECK-ENCODING: [0x0b,0x64,0x45,0x9e]
