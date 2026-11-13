# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vmslt.vv v0, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmslt.vv v0, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x00,0x4a,0x6c]

vmseq.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmseq.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x60]

vmseq.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmseq.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x62]

vmseq.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmseq.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x60]

vmseq.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmseq.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x62]

vmseq.vi v8, v4, 15, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmseq.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x60]

vmseq.vi v8, v4, 15 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmseq.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x62]

vmsne.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsne.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x64]

vmsne.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsne.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x66]

vmsne.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsne.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x64]

vmsne.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsne.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x66]

vmsne.vi v8, v4, 15, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsne.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x64]

vmsne.vi v8, v4, 15 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsne.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x66]

vmsltu.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsltu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x68]

vmsltu.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsltu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6a]

vmsltu.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsltu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x68]

vmsltu.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsltu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x6a]

vmslt.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmslt.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6c]

vmslt.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmslt.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6e]

vmslt.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmslt.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x6c]

vmslt.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmslt.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x6e]

vmsleu.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsleu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x70]

vmsleu.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsleu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x72]

vmsleu.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsleu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x70]

vmsleu.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsleu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x72]

vmsleu.vi v8, v4, 15, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsleu.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x70]

vmsleu.vi v8, v4, 15 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsleu.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x72]

vmsle.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsle.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x74]

vmsle.vv v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsle.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x76]

vmsle.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsle.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x74]

vmsle.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsle.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x76]

vmsle.vi v8, v4, 15, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsle.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x74]

vmsle.vi v8, v4, 15 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsle.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x76]

vmsgtu.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsgtu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x78]

vmsgtu.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsgtu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x7a]

vmsgtu.vi v8, v4, 15, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsgtu.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x78]

vmsgtu.vi v8, v4, 15 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsgtu.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7a]

vmsgt.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsgt.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x7c]

vmsgt.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsgt.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x7e]

vmsgt.vi v8, v4, 15, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsgt.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7c]

vmsgt.vi v8, v4, 15 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsgt.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7e]

vmsgtu.vv v8, v20, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsltu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x68]

vmsgtu.vv v8, v20, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsltu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6a]

vmsgt.vv v8, v20, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmslt.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6c]

vmsgt.vv v8, v20, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmslt.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6e]

vmsgeu.vv v8, v20, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsleu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x70]

vmsgeu.vv v8, v20, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsleu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x72]

vmsge.vv v8, v20, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsle.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x74]

vmsge.vv v8, v20, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsle.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x76]

vmsltu.vi v8, v4, 16, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsleu.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x70]

vmsltu.vi v8, v4, 16 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsleu.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x72]

vmsltu.vi v8, v4, 0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsne.vv v8, v4, v4, v0.t
# CHECK-ENCODING: [0x57,0x04,0x42,0x64]

vmsltu.vi v8, v4, 0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsne.vv v8, v4, v4
# CHECK-ENCODING: [0x57,0x04,0x42,0x66]

vmslt.vi v8, v4, 16, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsle.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x74]

vmslt.vi v8, v4, 16 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsle.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x76]

vmsgeu.vi v8, v4, 16, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsgtu.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x78]

vmsgeu.vi v8, v4, 16 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsgtu.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7a]

vmsgeu.vi v8, v4, 0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmseq.vv v8, v4, v4, v0.t
# CHECK-ENCODING: [0x57,0x04,0x42,0x60]

vmsgeu.vi v8, v4, 0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmseq.vv v8, v4, v4
# CHECK-ENCODING: [0x57,0x04,0x42,0x62]

vmsge.vi v8, v4, 16, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsgt.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7c]

vmsge.vi v8, v4, 16 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsgt.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7e]

vmsgeu.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsltu.vx v8, v4, a0
# CHECK-INST: vmnot.m v8, v8
# CHECK-ENCODING: [0x57,0x44,0x45,0x6a]
# CHECK-ENCODING: [0x57,0x24,0x84,0x76]

vmsge.vx v0, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmslt.vx v0, v4, a0
# CHECK-INST: vmnot.m v0, v0
# CHECK-ENCODING: [0x57,0x40,0x45,0x6e]
# CHECK-ENCODING: [0x57,0x20,0x00,0x76]

vmsge.vx v8, v4, a0 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmslt.vx v8, v4, a0
# CHECK-INST: vmnot.m v8, v8
# CHECK-ENCODING: [0x57,0x44,0x45,0x6e]
# CHECK-ENCODING: [0x57,0x24,0x84,0x76]

vmsgeu.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsltu.vx v8, v4, a0, v0.t
# CHECK-INST: vmxor.mm v8, v8, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x68]
# CHECK-ENCODING: [0x57,0x24,0x80,0x6e]

vmsge.vx v8, v4, a0, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmslt.vx v8, v4, a0, v0.t
# CHECK-INST: vmxor.mm v8, v8, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x6c]
# CHECK-ENCODING: [0x57,0x24,0x80,0x6e]

vmsgeu.vx v0, v4, a0, v0.t, v2 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsltu.vx v2, v4, a0
# CHECK-INST: vmandn.mm v0, v0, v2
# CHECK-ENCODING: [0x57,0x41,0x45,0x6a]
# CHECK-ENCODING: [0x57,0x20,0x01,0x62]

vmsge.vx v0, v4, a0, v0.t, v2 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmslt.vx v2, v4, a0
# CHECK-INST: vmandn.mm v0, v0, v2
# CHECK-ENCODING: [0x57,0x41,0x45,0x6e]
# CHECK-ENCODING: [0x57,0x20,0x01,0x62]

vmsgeu.vx v9, v4, a0, v0.t, v2 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmsltu.vx v2, v4, a0
# CHECK-INST: vmandn.mm v2, v0, v2
# CHECK-INST: vmandn.mm v9, v9, v0
# CHECK-INST: vmor.mm v9, v2, v9
# CHECK-ENCODING: [0x57,0x41,0x45,0x6a]
# CHECK-ENCODING: [0x57,0x21,0x01,0x62]
# CHECK-ENCODING: [0xd7,0x24,0x90,0x62]
# CHECK-ENCODING: [0xd7,0xa4,0x24,0x6a]

vmsge.vx v8, v4, a0, v0.t, v2 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmslt.vx v2, v4, a0
# CHECK-INST: vmandn.mm v2, v0, v2
# CHECK-INST: vmandn.mm v8, v8, v0
# CHECK-INST: vmor.mm v8, v2, v8
# CHECK-ENCODING: [0x57,0x41,0x45,0x6e]
# CHECK-ENCODING: [0x57,0x21,0x01,0x62]
# CHECK-ENCODING: [0x57,0x24,0x80,0x62]
# CHECK-ENCODING: [0x57,0x24,0x24,0x6a]
