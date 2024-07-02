# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vdivu.vv v8, v4, v20, v0.t
# CHECK-INST: vdivu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 804a2457 <unknown>

vdivu.vv v8, v4, v20
# CHECK-INST: vdivu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 824a2457 <unknown>

vdivu.vx v8, v4, a0, v0.t
# CHECK-INST: vdivu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 80456457 <unknown>

vdivu.vx v8, v4, a0
# CHECK-INST: vdivu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 82456457 <unknown>

vdiv.vv v8, v4, v20, v0.t
# CHECK-INST: vdiv.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x84]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 844a2457 <unknown>

vdiv.vv v8, v4, v20
# CHECK-INST: vdiv.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x86]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 864a2457 <unknown>

vdiv.vx v8, v4, a0, v0.t
# CHECK-INST: vdiv.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 84456457 <unknown>

vdiv.vx v8, v4, a0
# CHECK-INST: vdiv.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 86456457 <unknown>

vremu.vv v8, v4, v20, v0.t
# CHECK-INST: vremu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 884a2457 <unknown>

vremu.vv v8, v4, v20
# CHECK-INST: vremu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8a4a2457 <unknown>

vremu.vx v8, v4, a0, v0.t
# CHECK-INST: vremu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 88456457 <unknown>

vremu.vx v8, v4, a0
# CHECK-INST: vremu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8a456457 <unknown>

vrem.vv v8, v4, v20, v0.t
# CHECK-INST: vrem.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x8c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8c4a2457 <unknown>

vrem.vv v8, v4, v20
# CHECK-INST: vrem.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x8e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8e4a2457 <unknown>

vrem.vx v8, v4, a0, v0.t
# CHECK-INST: vrem.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8c456457 <unknown>

vrem.vx v8, v4, a0
# CHECK-INST: vrem.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8e456457 <unknown>
