# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vnclipu.wv v8, v4, v20, v0.t
# CHECK-INST: vnclipu.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xb8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b84a0457 <unknown>

vnclipu.wv v8, v4, v20
# CHECK-INST: vnclipu.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xba]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ba4a0457 <unknown>

vnclipu.wx v8, v4, a0, v0.t
# CHECK-INST: vnclipu.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xb8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b8454457 <unknown>

vnclipu.wx v8, v4, a0
# CHECK-INST: vnclipu.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xba]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ba454457 <unknown>

vnclipu.wi v8, v4, 31, v0.t
# CHECK-INST: vnclipu.wi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xb8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b84fb457 <unknown>

vnclipu.wi v8, v4, 31
# CHECK-INST: vnclipu.wi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xba]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ba4fb457 <unknown>

vnclip.wv v8, v4, v20, v0.t
# CHECK-INST: vnclip.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xbc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: bc4a0457 <unknown>

vnclip.wv v8, v4, v20
# CHECK-INST: vnclip.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xbe]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: be4a0457 <unknown>

vnclip.wx v8, v4, a0, v0.t
# CHECK-INST: vnclip.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: bc454457 <unknown>

vnclip.wx v8, v4, a0
# CHECK-INST: vnclip.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: be454457 <unknown>

vnclip.wi v8, v4, 31, v0.t
# CHECK-INST: vnclip.wi v8, v4, 31, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xbc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: bc4fb457 <unknown>

vnclip.wi v8, v4, 31
# CHECK-INST: vnclip.wi v8, v4, 31
# CHECK-ENCODING: [0x57,0xb4,0x4f,0xbe]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: be4fb457 <unknown>
