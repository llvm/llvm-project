# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vand.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vand.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x24]

vand.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vand.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x26]

vand.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vand.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x24]

vand.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vand.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x26]

vand.vi v8, v4, 15, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vand.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x24]

vand.vi v8, v4, 15 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vand.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x26]
