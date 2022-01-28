# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vor.vv v8, v4, v20, v0.t
# CHECK-INST: vor.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x28]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 57 04 4a 28 <unknown>

vor.vv v8, v4, v20
# CHECK-INST: vor.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x2a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 57 04 4a 2a <unknown>

vor.vx v8, v4, a0, v0.t
# CHECK-INST: vor.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x28]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 57 44 45 28 <unknown>

vor.vx v8, v4, a0
# CHECK-INST: vor.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x2a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 57 44 45 2a <unknown>

vor.vi v8, v4, 15, v0.t
# CHECK-INST: vor.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x28]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 57 b4 47 28 <unknown>

vor.vi v8, v4, 15
# CHECK-INST: vor.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x2a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 57 b4 47 2a <unknown>
