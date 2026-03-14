# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vmslt.vv v0, v4, v20, v0.t
# CHECK-INST: vmslt.vv v0, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x00,0x4a,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c4a0057 <unknown>

vmseq.vv v8, v4, v20, v0.t
# CHECK-INST: vmseq.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x60]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 604a0457 <unknown>

vmseq.vv v8, v4, v20
# CHECK-INST: vmseq.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 624a0457 <unknown>

vmseq.vx v8, v4, a0, v0.t
# CHECK-INST: vmseq.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x60]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 60454457 <unknown>

vmseq.vx v8, v4, a0
# CHECK-INST: vmseq.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 62454457 <unknown>

vmseq.vi v8, v4, 15, v0.t
# CHECK-INST: vmseq.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x60]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6047b457 <unknown>

vmseq.vi v8, v4, 15
# CHECK-INST: vmseq.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6247b457 <unknown>

vmsne.vv v8, v4, v20, v0.t
# CHECK-INST: vmsne.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x64]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 644a0457 <unknown>

vmsne.vv v8, v4, v20
# CHECK-INST: vmsne.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 664a0457 <unknown>

vmsne.vx v8, v4, a0, v0.t
# CHECK-INST: vmsne.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 64454457 <unknown>

vmsne.vx v8, v4, a0
# CHECK-INST: vmsne.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 66454457 <unknown>

vmsne.vi v8, v4, 15, v0.t
# CHECK-INST: vmsne.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x64]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6447b457 <unknown>

vmsne.vi v8, v4, 15
# CHECK-INST: vmsne.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6647b457 <unknown>

vmsltu.vv v8, v4, v20, v0.t
# CHECK-INST: vmsltu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x68]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 684a0457 <unknown>

vmsltu.vv v8, v4, v20
# CHECK-INST: vmsltu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6a4a0457 <unknown>

vmsltu.vx v8, v4, a0, v0.t
# CHECK-INST: vmsltu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x68]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 68454457 <unknown>

vmsltu.vx v8, v4, a0
# CHECK-INST: vmsltu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6a454457 <unknown>

vmslt.vv v8, v4, v20, v0.t
# CHECK-INST: vmslt.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c4a0457 <unknown>

vmslt.vv v8, v4, v20
# CHECK-INST: vmslt.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e4a0457 <unknown>

vmslt.vx v8, v4, a0, v0.t
# CHECK-INST: vmslt.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c454457 <unknown>

vmslt.vx v8, v4, a0
# CHECK-INST: vmslt.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e454457 <unknown>

vmsleu.vv v8, v4, v20, v0.t
# CHECK-INST: vmsleu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x70]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 704a0457 <unknown>

vmsleu.vv v8, v4, v20
# CHECK-INST: vmsleu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x72]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 724a0457 <unknown>

vmsleu.vx v8, v4, a0, v0.t
# CHECK-INST: vmsleu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x70]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 70454457 <unknown>

vmsleu.vx v8, v4, a0
# CHECK-INST: vmsleu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x72]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 72454457 <unknown>

vmsleu.vi v8, v4, 15, v0.t
# CHECK-INST: vmsleu.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x70]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7047b457 <unknown>

vmsleu.vi v8, v4, 15
# CHECK-INST: vmsleu.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x72]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7247b457 <unknown>

vmsle.vv v8, v4, v20, v0.t
# CHECK-INST: vmsle.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x74]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 744a0457 <unknown>

vmsle.vv v8, v4, v20
# CHECK-INST: vmsle.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 764a0457 <unknown>

vmsle.vx v8, v4, a0, v0.t
# CHECK-INST: vmsle.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x74]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 74454457 <unknown>

vmsle.vx v8, v4, a0
# CHECK-INST: vmsle.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 76454457 <unknown>

vmsle.vi v8, v4, 15, v0.t
# CHECK-INST: vmsle.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x74]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7447b457 <unknown>

vmsle.vi v8, v4, 15
# CHECK-INST: vmsle.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7647b457 <unknown>

vmsgtu.vx v8, v4, a0, v0.t
# CHECK-INST: vmsgtu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x78]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 78454457 <unknown>

vmsgtu.vx v8, v4, a0
# CHECK-INST: vmsgtu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x7a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7a454457 <unknown>

vmsgtu.vi v8, v4, 15, v0.t
# CHECK-INST: vmsgtu.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x78]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7847b457 <unknown>

vmsgtu.vi v8, v4, 15
# CHECK-INST: vmsgtu.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7a47b457 <unknown>

vmsgt.vx v8, v4, a0, v0.t
# CHECK-INST: vmsgt.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x7c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7c454457 <unknown>

vmsgt.vx v8, v4, a0
# CHECK-INST: vmsgt.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x7e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7e454457 <unknown>

vmsgt.vi v8, v4, 15, v0.t
# CHECK-INST: vmsgt.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7c47b457 <unknown>

vmsgt.vi v8, v4, 15
# CHECK-INST: vmsgt.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7e47b457 <unknown>

vmsgtu.vv v8, v20, v4, v0.t
# CHECK-INST: vmsltu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x68]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 684a0457 <unknown>

vmsgtu.vv v8, v20, v4
# CHECK-INST: vmsltu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6a4a0457 <unknown>

vmsgt.vv v8, v20, v4, v0.t
# CHECK-INST: vmslt.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c4a0457 <unknown>

vmsgt.vv v8, v20, v4
# CHECK-INST: vmslt.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e4a0457 <unknown>

vmsgeu.vv v8, v20, v4, v0.t
# CHECK-INST: vmsleu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x70]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 704a0457 <unknown>

vmsgeu.vv v8, v20, v4
# CHECK-INST: vmsleu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x72]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 724a0457 <unknown>

vmsge.vv v8, v20, v4, v0.t
# CHECK-INST: vmsle.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x74]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 744a0457 <unknown>

vmsge.vv v8, v20, v4
# CHECK-INST: vmsle.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 764a0457 <unknown>

vmsltu.vi v8, v4, 16, v0.t
# CHECK-INST: vmsleu.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x70]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7047b457 <unknown>

vmsltu.vi v8, v4, 16
# CHECK-INST: vmsleu.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x72]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7247b457 <unknown>

vmsltu.vi v8, v4, 0, v0.t
# CHECK-INST: vmsne.vv v8, v4, v4, v0.t
# CHECK-ENCODING: [0x57,0x04,0x42,0x64]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 64420457 <unknown>

vmsltu.vi v8, v4, 0
# CHECK-INST: vmsne.vv v8, v4, v4
# CHECK-ENCODING: [0x57,0x04,0x42,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 66420457 <unknown>

vmslt.vi v8, v4, 16, v0.t
# CHECK-INST: vmsle.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x74]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7447b457 <unknown>

vmslt.vi v8, v4, 16
# CHECK-INST: vmsle.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7647b457 <unknown>

vmsgeu.vi v8, v4, 16, v0.t
# CHECK-INST: vmsgtu.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x78]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7847b457 <unknown>

vmsgeu.vi v8, v4, 16
# CHECK-INST: vmsgtu.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7a47b457 <unknown>

vmsgeu.vi v8, v4, 0, v0.t
# CHECK-INST: vmseq.vv v8, v4, v4, v0.t
# CHECK-ENCODING: [0x57,0x04,0x42,0x60]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 60420457 <unknown>

vmsgeu.vi v8, v4, 0
# CHECK-INST: vmseq.vv v8, v4, v4
# CHECK-ENCODING: [0x57,0x04,0x42,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 62420457 <unknown>

vmsge.vi v8, v4, 16, v0.t
# CHECK-INST: vmsgt.vi v8, v4, 0xf, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7c47b457 <unknown>

vmsge.vi v8, v4, 16
# CHECK-INST: vmsgt.vi v8, v4, 0xf
# CHECK-ENCODING: [0x57,0xb4,0x47,0x7e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7e47b457 <unknown>

vmsgeu.vx v8, v4, a0
# CHECK-INST: vmsltu.vx v8, v4, a0
# CHECK-INST: vmnot.m v8, v8
# CHECK-ENCODING: [0x57,0x44,0x45,0x6a]
# CHECK-ENCODING: [0x57,0x24,0x84,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6a454457 <unknown>
# CHECK-UNKNOWN: 76842457 <unknown>

vmsge.vx v0, v4, a0
# CHECK-INST: vmslt.vx v0, v4, a0
# CHECK-INST: vmnot.m v0, v0
# CHECK-ENCODING: [0x57,0x40,0x45,0x6e]
# CHECK-ENCODING: [0x57,0x20,0x00,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e454057 <unknown>
# CHECK-UNKNOWN: 76002057 <unknown>

vmsge.vx v8, v4, a0
# CHECK-INST: vmslt.vx v8, v4, a0
# CHECK-INST: vmnot.m v8, v8
# CHECK-ENCODING: [0x57,0x44,0x45,0x6e]
# CHECK-ENCODING: [0x57,0x24,0x84,0x76]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e454457 <unknown>
# CHECK-UNKNOWN: 76842457 <unknown>

vmsgeu.vx v8, v4, a0, v0.t
# CHECK-INST: vmsltu.vx v8, v4, a0, v0.t
# CHECK-INST: vmxor.mm v8, v8, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x68]
# CHECK-ENCODING: [0x57,0x24,0x80,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 68454457 <unknown>
# CHECK-UNKNOWN: 6e802457 <unknown>

vmsge.vx v8, v4, a0, v0.t
# CHECK-INST: vmslt.vx v8, v4, a0, v0.t
# CHECK-INST: vmxor.mm v8, v8, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x6c]
# CHECK-ENCODING: [0x57,0x24,0x80,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c454457 <unknown>
# CHECK-UNKNOWN: 6e802457 <unknown>

vmsgeu.vx v0, v4, a0, v0.t, v2
# CHECK-INST: vmsltu.vx v2, v4, a0
# CHECK-INST: vmandn.mm v0, v0, v2
# CHECK-ENCODING: [0x57,0x41,0x45,0x6a]
# CHECK-ENCODING: [0x57,0x20,0x01,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6a454157 <unknown>
# CHECK-UNKNOWN: 62012057 <unknown>

vmsge.vx v0, v4, a0, v0.t, v2
# CHECK-INST: vmslt.vx v2, v4, a0
# CHECK-INST: vmandn.mm v0, v0, v2
# CHECK-ENCODING: [0x57,0x41,0x45,0x6e]
# CHECK-ENCODING: [0x57,0x20,0x01,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e454157 <unknown>
# CHECK-UNKNOWN: 62012057 <unknown>

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
# CHECK-UNKNOWN: 6a454157 <unknown>
# CHECK-UNKNOWN: 62012157 <unknown>
# CHECK-UNKNOWN: 629024d7 <unknown>
# CHECK-UNKNOWN: 6a24a4d7 <unknown>

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
# CHECK-UNKNOWN: 6e454157 <unknown>
# CHECK-UNKNOWN: 62012157 <unknown>
# CHECK-UNKNOWN: 62802457 <unknown>
# CHECK-UNKNOWN: 6a242457 <unknown>
