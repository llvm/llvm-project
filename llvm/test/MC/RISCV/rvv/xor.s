# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vxor.vv v8, v4, v20, v0.t
# CHECK-INST: vxor.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x2c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2c4a0457 <unknown>

vxor.vv v8, v4, v20
# CHECK-INST: vxor.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x2e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2e4a0457 <unknown>

vxor.vx v8, v4, a0, v0.t
# CHECK-INST: vxor.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2c454457 <unknown>

vxor.vx v8, v4, a0
# CHECK-INST: vxor.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2e454457 <unknown>

vxor.vi v8, v4, 15, v0.t
# CHECK-INST: vxor.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x2c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2c47b457 <unknown>

vxor.vi v8, v4, 15
# CHECK-INST: vxor.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x2e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2e47b457 <unknown>

vnot.v v8, v4, v0.t
# CHECK-INST: vnot.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x2c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2c4fb457 <unknown>

vnot.v v8, v4
# CHECK-INST: vnot.v v8, v4
# CHECK-ENCODING: [0x57,0xb4,0x4f,0x2e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2e4fb457 <unknown>
