# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vsub.vv v8, v4, v20, v0.t
# CHECK-INST: vsub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsub.vv v8, v4, v20
# CHECK-INST: vsub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsub.vx v8, v4, a0, v0.t
# CHECK-INST: vsub.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsub.vx v8, v4, a0
# CHECK-INST: vsub.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vrsub.vx v8, v4, a0, v0.t
# CHECK-INST: vrsub.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vrsub.vx v8, v4, a0
# CHECK-INST: vrsub.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vrsub.vi v8, v4, 15, v0.t
# CHECK-INST: vrsub.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vrsub.vi v8, v4, 15
# CHECK-INST: vrsub.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwsubu.vv v8, v4, v20, v0.t
# CHECK-INST: vwsubu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xc8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwsubu.vv v8, v4, v20
# CHECK-INST: vwsubu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xca]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwsubu.vx v8, v4, a0, v0.t
# CHECK-INST: vwsubu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xc8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwsubu.vx v8, v4, a0
# CHECK-INST: vwsubu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xca]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwsub.vv v8, v4, v20, v0.t
# CHECK-INST: vwsub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xcc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwsub.vv v8, v4, v20
# CHECK-INST: vwsub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xce]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwsub.vx v8, v4, a0, v0.t
# CHECK-INST: vwsub.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwsub.vx v8, v4, a0
# CHECK-INST: vwsub.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwsubu.wv v8, v4, v20, v0.t
# CHECK-INST: vwsubu.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xd8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwsubu.wv v8, v4, v20
# CHECK-INST: vwsubu.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xda]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwsubu.wx v8, v4, a0, v0.t
# CHECK-INST: vwsubu.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xd8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwsubu.wx v8, v4, a0
# CHECK-INST: vwsubu.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xda]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwsub.wv v8, v4, v20, v0.t
# CHECK-INST: vwsub.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xdc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwsub.wv v8, v4, v20
# CHECK-INST: vwsub.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xde]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwsub.wx v8, v4, a0, v0.t
# CHECK-INST: vwsub.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xdc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwsub.wx v8, v4, a0
# CHECK-INST: vwsub.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xde]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsbc.vvm v8, v4, v20, v0
# CHECK-INST: vsbc.vvm v8, v4, v20, v0
# CHECK-ENCODING: [0x57,0x04,0x4a,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsbc.vvm v4, v4, v20, v0
# CHECK-INST: vsbc.vvm v4, v4, v20, v0
# CHECK-ENCODING: [0x57,0x02,0x4a,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsbc.vvm v8, v4, v8, v0
# CHECK-INST: vsbc.vvm v8, v4, v8, v0
# CHECK-ENCODING: [0x57,0x04,0x44,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsbc.vxm v8, v4, a0, v0
# CHECK-INST: vsbc.vxm v8, v4, a0, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsbc.vvm v8, v4, v20, v0
# CHECK-INST: vmsbc.vvm v8, v4, v20, v0
# CHECK-ENCODING: [0x57,0x04,0x4a,0x4c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsbc.vvm v4, v4, v20, v0
# CHECK-INST: vmsbc.vvm v4, v4, v20, v0
# CHECK-ENCODING: [0x57,0x02,0x4a,0x4c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsbc.vvm v8, v4, v8, v0
# CHECK-INST: vmsbc.vvm v8, v4, v8, v0
# CHECK-ENCODING: [0x57,0x04,0x44,0x4c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsbc.vxm v8, v4, a0, v0
# CHECK-INST: vmsbc.vxm v8, v4, a0, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsbc.vv v8, v4, v20
# CHECK-INST: vmsbc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x4e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmsbc.vx v8, v4, a0
# CHECK-INST: vmsbc.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vssubu.vv v8, v4, v20, v0.t
# CHECK-INST: vssubu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vssubu.vv v8, v4, v20
# CHECK-INST: vssubu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vssubu.vx v8, v4, a0, v0.t
# CHECK-INST: vssubu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vssubu.vx v8, v4, a0
# CHECK-INST: vssubu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vssub.vv v8, v4, v20, v0.t
# CHECK-INST: vssub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x8c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vssub.vv v8, v4, v20
# CHECK-INST: vssub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x8e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vssub.vx v8, v4, a0, v0.t
# CHECK-INST: vssub.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vssub.vx v8, v4, a0
# CHECK-INST: vssub.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vasub.vv v8, v4, v20, v0.t
# CHECK-INST: vasub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x2c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vasub.vv v8, v4, v20
# CHECK-INST: vasub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x2e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vasub.vx v8, v4, a0, v0.t
# CHECK-INST: vasub.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vasub.vx v8, v4, a0
# CHECK-INST: vasub.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vasubu.vv v8, v4, v20, v0.t
# CHECK-INST: vasubu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x28]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vasubu.vv v8, v4, v20
# CHECK-INST: vasubu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x2a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vasubu.vx v8, v4, a0, v0.t
# CHECK-INST: vasubu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x28]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vasubu.vx v8, v4, a0
# CHECK-INST: vasubu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x2a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
