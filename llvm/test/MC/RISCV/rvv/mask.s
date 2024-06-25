# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vmand.mm v8, v4, v20
# CHECK-INST: vmand.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 664a2457 <unknown>

vmnand.mm v8, v4, v20
# CHECK-INST: vmnand.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 764a2457 <unknown>

vmandn.mm v8, v4, v20
# CHECK-INST: vmandn.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 624a2457 <unknown>

vmxor.mm v8, v4, v20
# CHECK-INST: vmxor.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e4a2457 <unknown>

vmor.mm v8, v4, v20
# CHECK-INST: vmor.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6a4a2457 <unknown>

vmnor.mm v8, v4, v20
# CHECK-INST: vmnor.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x7a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7a4a2457 <unknown>

vmorn.mm v8, v4, v20
# CHECK-INST: vmorn.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x72]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 724a2457 <unknown>

vmxnor.mm v8, v4, v20
# CHECK-INST: vmxnor.mm v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x7e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7e4a2457 <unknown>

vcpop.m a2, v4, v0.t
# CHECK-INST: vcpop.m a2, v4, v0.t
# CHECK-ENCODING: [0x57,0x26,0x48,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 40482657 <unknown>

vcpop.m a2, v4
# CHECK-INST: vcpop.m a2, v4
# CHECK-ENCODING: [0x57,0x26,0x48,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 42482657 <unknown>

vfirst.m a2, v4, v0.t
# CHECK-INST: vfirst.m a2, v4, v0.t
# CHECK-ENCODING: [0x57,0xa6,0x48,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4048a657 <unknown>

vfirst.m a2, v4
# CHECK-INST: vfirst.m a2, v4
# CHECK-ENCODING: [0x57,0xa6,0x48,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4248a657 <unknown>

vmsbf.m v8, v4, v0.t
# CHECK-INST: vmsbf.m v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x40,0x50]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 5040a457 <unknown>

vmsbf.m v8, v4
# CHECK-INST: vmsbf.m v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x40,0x52]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 5240a457 <unknown>

vmsif.m v8, v4, v0.t
# CHECK-INST: vmsif.m v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x41,0x50]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 5041a457 <unknown>

vmsif.m v8, v4
# CHECK-INST: vmsif.m v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x41,0x52]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 5241a457 <unknown>

vmsof.m v8, v4, v0.t
# CHECK-INST: vmsof.m v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x41,0x50]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 50412457 <unknown>

vmsof.m v8, v4
# CHECK-INST: vmsof.m v8, v4
# CHECK-ENCODING: [0x57,0x24,0x41,0x52]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 52412457 <unknown>

viota.m v8, v4, v0.t
# CHECK-INST: viota.m v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x48,0x50]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 50482457 <unknown>

viota.m v8, v4
# CHECK-INST: viota.m v8, v4
# CHECK-ENCODING: [0x57,0x24,0x48,0x52]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 52482457 <unknown>

vid.v v8, v0.t
# CHECK-INST: vid.v v8, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x08,0x50]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 5008a457 <unknown>

vid.v v8
# CHECK-INST: vid.v v8
# CHECK-ENCODING: [0x57,0xa4,0x08,0x52]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 5208a457 <unknown>

vmmv.m v8, v4
# CHECK-INST: vmmv.m v8, v4
# CHECK-ENCODING: [0x57,0x24,0x42,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 66422457 <unknown>

vmclr.m v8
# CHECK-INST: vmclr.m v8
# CHECK-ENCODING: [0x57,0x24,0x84,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e842457 <unknown>

vmset.m v8
# CHECK-INST: vmset.m v8
# CHECK-ENCODING: [0x57,0x24,0x84,0x7e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7e842457 <unknown>

vmnot.m v8, v4
# CHECK-INST: vmnot.m v8, v4
# CHECK-ENCODING: [0x57,0x24,0x42,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 76422457 <unknown>
