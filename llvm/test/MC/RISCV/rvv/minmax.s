# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vminu.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vminu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x10]

vminu.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vminu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x12]

vminu.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vminu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x10]

vminu.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vminu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x12]

vmin.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmin.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x14]

vmin.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmin.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x16]

vmin.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmin.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x14]

vmin.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmin.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x16]

vmaxu.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmaxu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x18]

vmaxu.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmaxu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x1a]

vmaxu.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmaxu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x18]

vmaxu.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmaxu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x1a]

vmax.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmax.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x1c]

vmax.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmax.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x1e]

vmax.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmax.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x1c]

vmax.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmax.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x1e]
