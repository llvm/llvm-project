# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vmv.v.v v8, v20
# CHECK-INST: vmv.v.v v8, v20
# CHECK-ENCODING: [0x57,0x04,0x0a,0x5e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 57 04 0a 5e <unknown>

vmv.v.x v8, a0
# CHECK-INST: vmv.v.x v8, a0
# CHECK-ENCODING: [0x57,0x44,0x05,0x5e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 57 44 05 5e <unknown>

vmv.v.i v8, 15
# CHECK-INST: vmv.v.i v8, 15
# CHECK-ENCODING: [0x57,0xb4,0x07,0x5e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 57 b4 07 5e <unknown>

vmv.x.s a2, v4
# CHECK-INST: vmv.x.s a2, v4
# CHECK-ENCODING: [0x57,0x26,0x40,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 57 26 40 42 <unknown>

vmv.s.x v8, a0
# CHECK-INST: vmv.s.x v8, a0
# CHECK-ENCODING: [0x57,0x64,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 57 64 05 42 <unknown>

vmv1r.v v8, v4
# CHECK-INST: vmv1r.v v8, v4
# CHECK-ENCODING: [0x57,0x34,0x40,0x9e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 57 34 40 9e <unknown>

vmv2r.v v8, v4
# CHECK-INST: vmv2r.v v8, v4
# CHECK-ENCODING: [0x57,0xb4,0x40,0x9e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 57 b4 40 9e <unknown>

vmv4r.v v8, v4
# CHECK-INST: vmv4r.v v8, v4
# CHECK-ENCODING: [0x57,0xb4,0x41,0x9e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 57 b4 41 9e <unknown>

vmv8r.v v8, v24
# CHECK-INST: vmv8r.v v8, v24
# CHECK-ENCODING: [0x57,0xb4,0x83,0x9f]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors)
# CHECK-UNKNOWN: 57 b4 83 9f <unknown>
