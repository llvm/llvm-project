# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vdivu.vv v8, v4, v20, v0.t
# CHECK-INST: vdivu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vdivu.vv v8, v4, v20
# CHECK-INST: vdivu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vdivu.vx v8, v4, a0, v0.t
# CHECK-INST: vdivu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vdivu.vx v8, v4, a0
# CHECK-INST: vdivu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vdiv.vv v8, v4, v20, v0.t
# CHECK-INST: vdiv.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x84]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vdiv.vv v8, v4, v20
# CHECK-INST: vdiv.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x86]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vdiv.vx v8, v4, a0, v0.t
# CHECK-INST: vdiv.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vdiv.vx v8, v4, a0
# CHECK-INST: vdiv.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vremu.vv v8, v4, v20, v0.t
# CHECK-INST: vremu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vremu.vv v8, v4, v20
# CHECK-INST: vremu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vremu.vx v8, v4, a0, v0.t
# CHECK-INST: vremu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vremu.vx v8, v4, a0
# CHECK-INST: vremu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vrem.vv v8, v4, v20, v0.t
# CHECK-INST: vrem.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x8c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vrem.vv v8, v4, v20
# CHECK-INST: vrem.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x8e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vrem.vx v8, v4, a0, v0.t
# CHECK-INST: vrem.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vrem.vx v8, v4, a0
# CHECK-INST: vrem.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
