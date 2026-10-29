# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vmv.v.v v8, v20
# CHECK-INST: vmv.v.v v8, v20
# CHECK-ENCODING: [0x57,0x04,0x0a,0x5e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmv.v.x v8, a0
# CHECK-INST: vmv.v.x v8, a0
# CHECK-ENCODING: [0x57,0x44,0x05,0x5e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmv.v.i v8, 15
# CHECK-INST: vmv.v.i v8, 15
# CHECK-ENCODING: [0x57,0xb4,0x07,0x5e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmv.x.s a2, v4
# CHECK-INST: vmv.x.s a2, v4
# CHECK-ENCODING: [0x57,0x26,0x40,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmv.s.x v8, a0
# CHECK-INST: vmv.s.x v8, a0
# CHECK-ENCODING: [0x57,0x64,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmv1r.v v8, v4
# CHECK-INST: vmv1r.v v8, v4
# CHECK-ENCODING: [0x57,0x34,0x40,0x9e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmv2r.v v8, v4
# CHECK-INST: vmv2r.v v8, v4
# CHECK-ENCODING: [0x57,0xb4,0x40,0x9e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmv4r.v v8, v4
# CHECK-INST: vmv4r.v v8, v4
# CHECK-ENCODING: [0x57,0xb4,0x41,0x9e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmv8r.v v8, v24
# CHECK-INST: vmv8r.v v8, v24
# CHECK-ENCODING: [0x57,0xb4,0x83,0x9f]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
