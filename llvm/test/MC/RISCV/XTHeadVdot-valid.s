# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xtheadvdot %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xtheadvdot %s \
# RUN:        | llvm-objdump -d --mattr=+xtheadvdot - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

th.vmaqau.vv v8, v20, v4, v0.t
# CHECK-INST: th.vmaqau.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x0b,0x64,0x4a,0x88]
# expected-error@-3 {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}

th.vmaqau.vv v8, v20, v4
# CHECK-INST: th.vmaqau.vv v8, v20, v4
# CHECK-ENCODING: [0x0b,0x64,0x4a,0x8a]
# expected-error@-3 {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}

th.vmaqau.vx v8, a0, v4, v0.t
# CHECK-INST: th.vmaqau.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x0b,0x64,0x45,0x8c]
# expected-error@-3 {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}

th.vmaqau.vx v8, a0, v4
# CHECK-INST: th.vmaqau.vx v8, a0, v4
# CHECK-ENCODING: [0x0b,0x64,0x45,0x8e]
# expected-error@-3 {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}

th.vmaqa.vv v8, v20, v4, v0.t
# CHECK-INST: th.vmaqa.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x0b,0x64,0x4a,0x80]
# expected-error@-3 {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}

th.vmaqa.vv v8, v20, v4
# CHECK-INST: th.vmaqa.vv v8, v20, v4
# CHECK-ENCODING: [0x0b,0x64,0x4a,0x82]
# expected-error@-3 {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}

th.vmaqa.vx v8, a0, v4, v0.t
# CHECK-INST: th.vmaqa.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x0b,0x64,0x45,0x84]
# expected-error@-3 {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}

th.vmaqa.vx v8, a0, v4
# CHECK-INST: th.vmaqa.vx v8, a0, v4
# CHECK-ENCODING: [0x0b,0x64,0x45,0x86]
# expected-error@-3 {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}

th.vmaqasu.vv v8, v20, v4, v0.t
# CHECK-INST: th.vmaqasu.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x0b,0x64,0x4a,0x90]
# expected-error@-3 {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}

th.vmaqasu.vv v8, v20, v4
# CHECK-INST: th.vmaqasu.vv v8, v20, v4
# CHECK-ENCODING: [0x0b,0x64,0x4a,0x92]
# expected-error@-3 {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}

th.vmaqasu.vx v8, a0, v4, v0.t
# CHECK-INST: th.vmaqasu.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x0b,0x64,0x45,0x94]
# expected-error@-3 {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}

th.vmaqasu.vx v8, a0, v4
# CHECK-INST: th.vmaqasu.vx v8, a0, v4
# CHECK-ENCODING: [0x0b,0x64,0x45,0x96]
# expected-error@-3 {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}

th.vmaqaus.vx v8, a0, v4, v0.t
# CHECK-INST: th.vmaqaus.vx v8, a0, v4, v0.t
# CHECK-ENCODING: [0x0b,0x64,0x45,0x9c]
# expected-error@-3 {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}

th.vmaqaus.vx v8, a0, v4
# CHECK-INST: th.vmaqaus.vx v8, a0, v4
# CHECK-ENCODING: [0x0b,0x64,0x45,0x9e]
# expected-error@-3 {{instruction requires the following: 'XTHeadVdot' (T-Head Vector Extensions for Dot)}}
