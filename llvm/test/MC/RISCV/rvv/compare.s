# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vmslt.vv v0, v4, v20, v0.t
# CHECK-INST: vmslt.vv v0, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x00,0x4a,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmseq.vv v8, v4, v20, v0.t
# CHECK-INST: vmseq.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x60]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmseq.vv v8, v4, v20
# CHECK-INST: vmseq.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmseq.vx v8, v4, a0, v0.t
# CHECK-INST: vmseq.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x60]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmseq.vx v8, v4, a0
# CHECK-INST: vmseq.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmseq.vi v8, v4, 15, v0.t
# CHECK-INST: vmseq.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x60]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmseq.vi v8, v4, 15
# CHECK-INST: vmseq.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsne.vv v8, v4, v20, v0.t
# CHECK-INST: vmsne.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x64]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsne.vv v8, v4, v20
# CHECK-INST: vmsne.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsne.vx v8, v4, a0, v0.t
# CHECK-INST: vmsne.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsne.vx v8, v4, a0
# CHECK-INST: vmsne.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsne.vi v8, v4, 15, v0.t
# CHECK-INST: vmsne.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x64]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsne.vi v8, v4, 15
# CHECK-INST: vmsne.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsltu.vv v8, v4, v20, v0.t
# CHECK-INST: vmsltu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x68]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsltu.vv v8, v4, v20
# CHECK-INST: vmsltu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsltu.vx v8, v4, a0, v0.t
# CHECK-INST: vmsltu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x68]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsltu.vx v8, v4, a0
# CHECK-INST: vmsltu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmslt.vv v8, v4, v20, v0.t
# CHECK-INST: vmslt.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmslt.vv v8, v4, v20
# CHECK-INST: vmslt.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmslt.vx v8, v4, a0, v0.t
# CHECK-INST: vmslt.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmslt.vx v8, v4, a0
# CHECK-INST: vmslt.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsleu.vv v8, v4, v20, v0.t
# CHECK-INST: vmsleu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x70]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsleu.vv v8, v4, v20
# CHECK-INST: vmsleu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x72]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsleu.vx v8, v4, a0, v0.t
# CHECK-INST: vmsleu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x70]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsleu.vx v8, v4, a0
# CHECK-INST: vmsleu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x72]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsleu.vi v8, v4, 15, v0.t
# CHECK-INST: vmsleu.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x70]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsleu.vi v8, v4, 15
# CHECK-INST: vmsleu.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x72]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsle.vv v8, v4, v20, v0.t
# CHECK-INST: vmsle.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x74]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsle.vv v8, v4, v20
# CHECK-INST: vmsle.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsle.vx v8, v4, a0, v0.t
# CHECK-INST: vmsle.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x74]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsle.vx v8, v4, a0
# CHECK-INST: vmsle.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsle.vi v8, v4, 15, v0.t
# CHECK-INST: vmsle.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x74]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsle.vi v8, v4, 15
# CHECK-INST: vmsle.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgtu.vx v8, v4, a0, v0.t
# CHECK-INST: vmsgtu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x78]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgtu.vx v8, v4, a0
# CHECK-INST: vmsgtu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x7a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgtu.vi v8, v4, 15, v0.t
# CHECK-INST: vmsgtu.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x78]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgtu.vi v8, v4, 15
# CHECK-INST: vmsgtu.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgt.vx v8, v4, a0, v0.t
# CHECK-INST: vmsgt.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x7c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgt.vx v8, v4, a0
# CHECK-INST: vmsgt.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x7e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgt.vi v8, v4, 15, v0.t
# CHECK-INST: vmsgt.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgt.vi v8, v4, 15
# CHECK-INST: vmsgt.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgtu.vv v8, v20, v4, v0.t
# CHECK-INST: vmsltu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x68]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgtu.vv v8, v20, v4
# CHECK-INST: vmsltu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgt.vv v8, v20, v4, v0.t
# CHECK-INST: vmslt.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgt.vv v8, v20, v4
# CHECK-INST: vmslt.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgeu.vv v8, v20, v4, v0.t
# CHECK-INST: vmsleu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x70]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgeu.vv v8, v20, v4
# CHECK-INST: vmsleu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x72]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsge.vv v8, v20, v4, v0.t
# CHECK-INST: vmsle.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x74]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsge.vv v8, v20, v4
# CHECK-INST: vmsle.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsltu.vi v8, v4, 16, v0.t
# CHECK-INST: vmsleu.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x70]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsltu.vi v8, v4, 16
# CHECK-INST: vmsleu.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x72]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsltu.vi v8, v4, 0, v0.t
# CHECK-INST: vmsne.vv v8, v4, v4, v0.t
# CHECK-ENCODING: [0x57,0x04,0x42,0x64]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsltu.vi v8, v4, 0
# CHECK-INST: vmsne.vv v8, v4, v4
# CHECK-ENCODING: [0x57,0x04,0x42,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmslt.vi v8, v4, 16, v0.t
# CHECK-INST: vmsle.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x74]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmslt.vi v8, v4, 16
# CHECK-INST: vmsle.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgeu.vi v8, v4, 16, v0.t
# CHECK-INST: vmsgtu.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x78]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgeu.vi v8, v4, 16
# CHECK-INST: vmsgtu.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgeu.vi v8, v4, 0, v0.t
# CHECK-INST: vmseq.vv v8, v4, v4, v0.t
# CHECK-ENCODING: [0x57,0x04,0x42,0x60]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgeu.vi v8, v4, 0
# CHECK-INST: vmseq.vv v8, v4, v4
# CHECK-ENCODING: [0x57,0x04,0x42,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsge.vi v8, v4, 16, v0.t
# CHECK-INST: vmsgt.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsge.vi v8, v4, 16
# CHECK-INST: vmsgt.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgeu.vx v8, v4, a0
# CHECK-INST: vmsltu.vx v8, v4, a0
# CHECK-INST: vmnot.m v8, v8
# CHECK-ENCODING: [0x57,0x44,0x45,0x6a]
# CHECK-ENCODING: [0x57,0x24,0x84,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsge.vx v0, v4, a0
# CHECK-INST: vmslt.vx v0, v4, a0
# CHECK-INST: vmnot.m v0, v0
# CHECK-ENCODING: [0x57,0x40,0x45,0x6e]
# CHECK-ENCODING: [0x57,0x20,0x00,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsge.vx v8, v4, a0
# CHECK-INST: vmslt.vx v8, v4, a0
# CHECK-INST: vmnot.m v8, v8
# CHECK-ENCODING: [0x57,0x44,0x45,0x6e]
# CHECK-ENCODING: [0x57,0x24,0x84,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgeu.vx v8, v4, a0, v0.t
# CHECK-INST: vmsltu.vx v8, v4, a0, v0.t
# CHECK-INST: vmxor.mm v8, v8, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x68]
# CHECK-ENCODING: [0x57,0x24,0x80,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsge.vx v8, v4, a0, v0.t
# CHECK-INST: vmslt.vx v8, v4, a0, v0.t
# CHECK-INST: vmxor.mm v8, v8, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x6c]
# CHECK-ENCODING: [0x57,0x24,0x80,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgeu.vx v0, v4, a0, v0.t, v2
# CHECK-INST: vmsltu.vx v2, v4, a0
# CHECK-INST: vmandn.mm v0, v0, v2
# CHECK-ENCODING: [0x57,0x41,0x45,0x6a]
# CHECK-ENCODING: [0x57,0x20,0x01,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsge.vx v0, v4, a0, v0.t, v2
# CHECK-INST: vmslt.vx v2, v4, a0
# CHECK-INST: vmandn.mm v0, v0, v2
# CHECK-ENCODING: [0x57,0x41,0x45,0x6e]
# CHECK-ENCODING: [0x57,0x20,0x01,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsgeu.vx v9, v4, a0, v0.t, v2
# CHECK-INST: vmsltu.vx v2, v4, a0
# CHECK-INST: vmandn.mm v2, v0, v2
# CHECK-INST: vmandn.mm v9, v9, v0
# CHECK-INST: vmor.mm v9, v2, v9
# CHECK-ENCODING: [0x57,0x41,0x45,0x6a]
# CHECK-ENCODING: [0x57,0x21,0x01,0x62]
# CHECK-ENCODING: [0xd7,0x24,0x90,0x62]
# CHECK-ENCODING: [0xd7,0xa4,0x24,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsge.vx v8, v4, a0, v0.t, v2
# CHECK-INST: vmslt.vx v2, v4, a0
# CHECK-INST: vmandn.mm v2, v0, v2
# CHECK-INST: vmandn.mm v8, v8, v0
# CHECK-INST: vmor.mm v8, v2, v8
# CHECK-ENCODING: [0x57,0x41,0x45,0x6e]
# CHECK-ENCODING: [0x57,0x21,0x01,0x62]
# CHECK-ENCODING: [0x57,0x24,0x80,0x62]
# CHECK-ENCODING: [0x57,0x24,0x24,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
