# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vminu.vv v8, v4, v20, v0.t
# CHECK-INST: vminu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x10]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 04 4a 10 <unknown>

vminu.vv v8, v4, v20
# CHECK-INST: vminu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x12]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 04 4a 12 <unknown>

vminu.vx v8, v4, a0, v0.t
# CHECK-INST: vminu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x10]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 44 45 10 <unknown>

vminu.vx v8, v4, a0
# CHECK-INST: vminu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x12]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 44 45 12 <unknown>

vmin.vv v8, v4, v20, v0.t
# CHECK-INST: vmin.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x14]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 04 4a 14 <unknown>

vmin.vv v8, v4, v20
# CHECK-INST: vmin.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x16]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 04 4a 16 <unknown>

vmin.vx v8, v4, a0, v0.t
# CHECK-INST: vmin.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x14]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 44 45 14 <unknown>

vmin.vx v8, v4, a0
# CHECK-INST: vmin.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x16]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 44 45 16 <unknown>

vmaxu.vv v8, v4, v20, v0.t
# CHECK-INST: vmaxu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x18]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 04 4a 18 <unknown>

vmaxu.vv v8, v4, v20
# CHECK-INST: vmaxu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x1a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 04 4a 1a <unknown>

vmaxu.vx v8, v4, a0, v0.t
# CHECK-INST: vmaxu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x18]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 44 45 18 <unknown>

vmaxu.vx v8, v4, a0
# CHECK-INST: vmaxu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x1a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 44 45 1a <unknown>

vmax.vv v8, v4, v20, v0.t
# CHECK-INST: vmax.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 04 4a 1c <unknown>

vmax.vv v8, v4, v20
# CHECK-INST: vmax.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 04 4a 1e <unknown>

vmax.vx v8, v4, a0, v0.t
# CHECK-INST: vmax.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 44 45 1c <unknown>

vmax.vx v8, v4, a0
# CHECK-INST: vmax.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x1e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 44 45 1e <unknown>
