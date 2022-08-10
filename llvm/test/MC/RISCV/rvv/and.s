# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vand.vv v8, v4, v20, v0.t
# CHECK-INST: vand.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x24]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 04 4a 24 <unknown>

vand.vv v8, v4, v20
# CHECK-INST: vand.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x26]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 04 4a 26 <unknown>

vand.vx v8, v4, a0, v0.t
# CHECK-INST: vand.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 44 45 24 <unknown>

vand.vx v8, v4, a0
# CHECK-INST: vand.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 44 45 26 <unknown>

vand.vi v8, v4, 15, v0.t
# CHECK-INST: vand.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x24]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 b4 47 24 <unknown>

vand.vi v8, v4, 15
# CHECK-INST: vand.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x26]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 b4 47 26 <unknown>
