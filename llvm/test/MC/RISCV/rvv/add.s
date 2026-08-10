# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vadd.vv v8, v4, v20, v0.t
# CHECK-INST: vadd.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vadd.vv v8, v4, v20
# CHECK-INST: vadd.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vadd.vx v8, v4, a0, v0.t
# CHECK-INST: vadd.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vadd.vx v8, v4, a0
# CHECK-INST: vadd.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vadd.vi v8, v4, 15, v0.t
# CHECK-INST: vadd.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vadd.vi v8, v4, 15
# CHECK-INST: vadd.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwaddu.vv v8, v4, v20, v0.t
# CHECK-INST: vwaddu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xc0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwaddu.vv v8, v4, v20
# CHECK-INST: vwaddu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xc2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwaddu.vx v8, v4, a0, v0.t
# CHECK-INST: vwaddu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xc0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwaddu.vx v8, v4, a0
# CHECK-INST: vwaddu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xc2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwadd.vv v8, v4, v20, v0.t
# CHECK-INST: vwadd.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xc4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwadd.vv v8, v4, v20
# CHECK-INST: vwadd.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xc6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwadd.vx v8, v4, a0, v0.t
# CHECK-INST: vwadd.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwadd.vx v8, v4, a0
# CHECK-INST: vwadd.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwaddu.wv v8, v4, v20, v0.t
# CHECK-INST: vwaddu.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xd0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwaddu.wv v8, v4, v20
# CHECK-INST: vwaddu.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xd2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwaddu.wx v8, v4, a0, v0.t
# CHECK-INST: vwaddu.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xd0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwaddu.wx v8, v4, a0
# CHECK-INST: vwaddu.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xd2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwadd.wv v8, v4, v20, v0.t
# CHECK-INST: vwadd.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xd4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwadd.wv v8, v4, v20
# CHECK-INST: vwadd.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xd6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwadd.wx v8, v4, a0, v0.t
# CHECK-INST: vwadd.wx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xd4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwadd.wx v8, v4, a0
# CHECK-INST: vwadd.wx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xd6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vadc.vvm v8, v4, v20, v0
# CHECK-INST: vadc.vvm v8, v4, v20, v0
# CHECK-ENCODING: [0x57,0x04,0x4a,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vadc.vvm v4, v4, v20, v0
# CHECK-INST: vadc.vvm v4, v4, v20, v0
# CHECK-ENCODING: [0x57,0x02,0x4a,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vadc.vvm v8, v4, v8, v0
# CHECK-INST: vadc.vvm v8, v4, v8, v0
# CHECK-ENCODING: [0x57,0x04,0x44,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vadc.vxm v8, v4, a0, v0
# CHECK-INST: vadc.vxm v8, v4, a0, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vadc.vim v8, v4, 15, v0
# CHECK-INST: vadc.vim v8, v4, 15, v0
# CHECK-ENCODING: [0x57,0xb4,0x47,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmadc.vvm v8, v4, v20, v0
# CHECK-INST: vmadc.vvm v8, v4, v20, v0
# CHECK-ENCODING: [0x57,0x04,0x4a,0x44]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmadc.vvm v4, v4, v20, v0
# CHECK-INST: vmadc.vvm v4, v4, v20, v0
# CHECK-ENCODING: [0x57,0x02,0x4a,0x44]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmadc.vvm v8, v4, v8, v0
# CHECK-INST: vmadc.vvm v8, v4, v8, v0
# CHECK-ENCODING: [0x57,0x04,0x44,0x44]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmadc.vxm v8, v4, a0, v0
# CHECK-INST: vmadc.vxm v8, v4, a0, v0
# CHECK-ENCODING: [0x57,0x44,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmadc.vim v8, v4, 15, v0
# CHECK-INST: vmadc.vim v8, v4, 15, v0
# CHECK-ENCODING: [0x57,0xb4,0x47,0x44]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmadc.vv v8, v4, v20
# CHECK-INST: vmadc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x46]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmadc.vx v8, v4, a0
# CHECK-INST: vmadc.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmadc.vi v8, v4, 15
# CHECK-INST: vmadc.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x46]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsaddu.vv v8, v4, v20, v0.t
# CHECK-INST: vsaddu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsaddu.vv v8, v4, v20
# CHECK-INST: vsaddu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsaddu.vx v8, v4, a0, v0.t
# CHECK-INST: vsaddu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsaddu.vx v8, v4, a0
# CHECK-INST: vsaddu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsaddu.vi v8, v4, 15, v0.t
# CHECK-INST: vsaddu.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsaddu.vi v8, v4, 15
# CHECK-INST: vsaddu.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsadd.vv v8, v4, v20, v0.t
# CHECK-INST: vsadd.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x84]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsadd.vv v8, v4, v20
# CHECK-INST: vsadd.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x86]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsadd.vx v8, v4, a0, v0.t
# CHECK-INST: vsadd.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsadd.vx v8, v4, a0
# CHECK-INST: vsadd.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsadd.vi v8, v4, 15, v0.t
# CHECK-INST: vsadd.vi v8, v4, 15, v0.t
# CHECK-ENCODING: [0x57,0xb4,0x47,0x84]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsadd.vi v8, v4, 15
# CHECK-INST: vsadd.vi v8, v4, 15
# CHECK-ENCODING: [0x57,0xb4,0x47,0x86]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vaadd.vv v8, v4, v20, v0.t
# CHECK-INST: vaadd.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x24]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vaadd.vv v8, v4, v20
# CHECK-INST: vaadd.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x26]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vaadd.vx v8, v4, a0, v0.t
# CHECK-INST: vaadd.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vaadd.vx v8, v4, a0
# CHECK-INST: vaadd.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vaaddu.vv v8, v4, v20, v0.t
# CHECK-INST: vaaddu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x20]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vaaddu.vv v8, v4, v20
# CHECK-INST: vaaddu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vaaddu.vx v8, v4, a0, v0.t
# CHECK-INST: vaaddu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x20]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vaaddu.vx v8, v4, a0
# CHECK-INST: vaaddu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwcvt.x.x.v v8, v4, v0.t
# CHECK-INST: vwcvt.x.x.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x40,0xc4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwcvt.x.x.v v8, v4
# CHECK-INST: vwcvt.x.x.v v8, v4
# CHECK-ENCODING: [0x57,0x64,0x40,0xc6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwcvtu.x.x.v v8, v4, v0.t
# CHECK-INST: vwcvtu.x.x.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x40,0xc0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwcvtu.x.x.v v8, v4
# CHECK-INST: vwcvtu.x.x.v v8, v4
# CHECK-ENCODING: [0x57,0x64,0x40,0xc2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
