# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vmand.mm v8, v4, v20
# CHECK-INST: vmand.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmnand.mm v8, v4, v20
# CHECK-INST: vmnand.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmandn.mm v8, v4, v20
# CHECK-INST: vmandn.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmxor.mm v8, v4, v20
# CHECK-INST: vmxor.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmor.mm v8, v4, v20
# CHECK-INST: vmor.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmnor.mm v8, v4, v20
# CHECK-INST: vmnor.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x7a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmorn.mm v8, v4, v20
# CHECK-INST: vmorn.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x72]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmxnor.mm v8, v4, v20
# CHECK-INST: vmxnor.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x7e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vcpop.m a2, v4, v0.t
# CHECK-INST: vcpop.m a2, v4, v0.t
# CHECK-ENCODING: [0x57,0x26,0x48,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vcpop.m a2, v4
# CHECK-INST: vcpop.m a2, v4
# CHECK-ENCODING: [0x57,0x26,0x48,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vfirst.m a2, v4, v0.t
# CHECK-INST: vfirst.m a2, v4, v0.t
# CHECK-ENCODING: [0x57,0xa6,0x48,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vfirst.m a2, v4
# CHECK-INST: vfirst.m a2, v4
# CHECK-ENCODING: [0x57,0xa6,0x48,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsbf.m v8, v4, v0.t
# CHECK-INST: vmsbf.m v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x40,0x50]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsbf.m v8, v4
# CHECK-INST: vmsbf.m v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x40,0x52]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsif.m v8, v4, v0.t
# CHECK-INST: vmsif.m v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x41,0x50]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsif.m v8, v4
# CHECK-INST: vmsif.m v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x41,0x52]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsof.m v8, v4, v0.t
# CHECK-INST: vmsof.m v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x41,0x50]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsof.m v8, v4
# CHECK-INST: vmsof.m v8, v4
# CHECK-ENCODING: [0x57,0x24,0x41,0x52]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

viota.m v8, v4, v0.t
# CHECK-INST: viota.m v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x48,0x50]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

viota.m v8, v4
# CHECK-INST: viota.m v8, v4
# CHECK-ENCODING: [0x57,0x24,0x48,0x52]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vid.v v8, v0.t
# CHECK-INST: vid.v v8, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x08,0x50]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vid.v v8
# CHECK-INST: vid.v v8
# CHECK-ENCODING: [0x57,0xa4,0x08,0x52]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmmv.m v8, v4
# CHECK-INST: vmmv.m v8, v4
# CHECK-ENCODING: [0x57,0x24,0x42,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmclr.m v8
# CHECK-INST: vmclr.m v8
# CHECK-ENCODING: [0x57,0x24,0x84,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmset.m v8
# CHECK-INST: vmset.m v8
# CHECK-ENCODING: [0x57,0x24,0x84,0x7e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmnot.m v8, v4
# CHECK-INST: vmnot.m v8, v4
# CHECK-ENCODING: [0x57,0x24,0x42,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
