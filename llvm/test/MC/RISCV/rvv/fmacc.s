# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:         --mattr=+f \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:         --mattr=+f \
# RUN:        | llvm-objdump -d --mattr=+v --mattr=+f - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:         --mattr=+f \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vfmacc.vv v8, v20, v4, v0.t
# CHECK-INST: vfmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xb0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b04a1457 <unknown>

vfmacc.vv v8, v20, v4
# CHECK-INST: vfmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xb2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b24a1457 <unknown>

vfmacc.vf v8, fa0, v4, v0.t
# CHECK-INST: vfmacc.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xb0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b0455457 <unknown>

vfmacc.vf v8, fa0, v4
# CHECK-INST: vfmacc.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xb2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b2455457 <unknown>

vfnmacc.vv v8, v20, v4, v0.t
# CHECK-INST: vfnmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xb4]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b44a1457 <unknown>

vfnmacc.vv v8, v20, v4
# CHECK-INST: vfnmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xb6]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b64a1457 <unknown>

vfnmacc.vf v8, fa0, v4, v0.t
# CHECK-INST: vfnmacc.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xb4]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b4455457 <unknown>

vfnmacc.vf v8, fa0, v4
# CHECK-INST: vfnmacc.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xb6]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b6455457 <unknown>

vfmsac.vv v8, v20, v4, v0.t
# CHECK-INST: vfmsac.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xb8]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b84a1457 <unknown>

vfmsac.vv v8, v20, v4
# CHECK-INST: vfmsac.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xba]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ba4a1457 <unknown>

vfmsac.vf v8, fa0, v4, v0.t
# CHECK-INST: vfmsac.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xb8]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: b8455457 <unknown>

vfmsac.vf v8, fa0, v4
# CHECK-INST: vfmsac.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xba]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ba455457 <unknown>

vfnmsac.vv v8, v20, v4, v0.t
# CHECK-INST: vfnmsac.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xbc]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: bc4a1457 <unknown>

vfnmsac.vv v8, v20, v4
# CHECK-INST: vfnmsac.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xbe]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: be4a1457 <unknown>

vfnmsac.vf v8, fa0, v4, v0.t
# CHECK-INST: vfnmsac.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: bc455457 <unknown>

vfnmsac.vf v8, fa0, v4
# CHECK-INST: vfnmsac.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: be455457 <unknown>

vfmadd.vv v8, v20, v4, v0.t
# CHECK-INST: vfmadd.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xa0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a04a1457 <unknown>

vfmadd.vv v8, v20, v4
# CHECK-INST: vfmadd.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xa2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a24a1457 <unknown>

vfmadd.vf v8, fa0, v4, v0.t
# CHECK-INST: vfmadd.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xa0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a0455457 <unknown>

vfmadd.vf v8, fa0, v4
# CHECK-INST: vfmadd.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xa2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a2455457 <unknown>

vfnmadd.vv v8, v20, v4, v0.t
# CHECK-INST: vfnmadd.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xa4]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a44a1457 <unknown>

vfnmadd.vv v8, v20, v4
# CHECK-INST: vfnmadd.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xa6]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a64a1457 <unknown>

vfnmadd.vf v8, fa0, v4, v0.t
# CHECK-INST: vfnmadd.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a4455457 <unknown>

vfnmadd.vf v8, fa0, v4
# CHECK-INST: vfnmadd.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a6455457 <unknown>

vfmsub.vv v8, v20, v4, v0.t
# CHECK-INST: vfmsub.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xa8]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a84a1457 <unknown>

vfmsub.vv v8, v20, v4
# CHECK-INST: vfmsub.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xaa]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: aa4a1457 <unknown>

vfmsub.vf v8, fa0, v4, v0.t
# CHECK-INST: vfmsub.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xa8]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a8455457 <unknown>

vfmsub.vf v8, fa0, v4
# CHECK-INST: vfmsub.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xaa]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: aa455457 <unknown>

vfnmsub.vv v8, v20, v4, v0.t
# CHECK-INST: vfnmsub.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xac]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ac4a1457 <unknown>

vfnmsub.vv v8, v20, v4
# CHECK-INST: vfnmsub.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xae]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ae4a1457 <unknown>

vfnmsub.vf v8, fa0, v4, v0.t
# CHECK-INST: vfnmsub.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ac455457 <unknown>

vfnmsub.vf v8, fa0, v4
# CHECK-INST: vfnmsub.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ae455457 <unknown>

vfwmacc.vv v8, v20, v4, v0.t
# CHECK-INST: vfwmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xf0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: f04a1457 <unknown>

vfwmacc.vv v8, v20, v4
# CHECK-INST: vfwmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xf2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: f24a1457 <unknown>

vfwmacc.vf v8, fa0, v4, v0.t
# CHECK-INST: vfwmacc.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xf0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: f0455457 <unknown>

vfwmacc.vf v8, fa0, v4
# CHECK-INST: vfwmacc.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xf2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: f2455457 <unknown>

vfwnmacc.vv v8, v20, v4, v0.t
# CHECK-INST: vfwnmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xf4]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: f44a1457 <unknown>

vfwnmacc.vv v8, v20, v4
# CHECK-INST: vfwnmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xf6]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: f64a1457 <unknown>

vfwnmacc.vf v8, fa0, v4, v0.t
# CHECK-INST: vfwnmacc.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xf4]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: f4455457 <unknown>

vfwnmacc.vf v8, fa0, v4
# CHECK-INST: vfwnmacc.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xf6]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: f6455457 <unknown>

vfwmsac.vv v8, v20, v4, v0.t
# CHECK-INST: vfwmsac.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xf8]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: f84a1457 <unknown>

vfwmsac.vv v8, v20, v4
# CHECK-INST: vfwmsac.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xfa]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: fa4a1457 <unknown>

vfwmsac.vf v8, fa0, v4, v0.t
# CHECK-INST: vfwmsac.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xf8]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: f8455457 <unknown>

vfwmsac.vf v8, fa0, v4
# CHECK-INST: vfwmsac.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xfa]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: fa455457 <unknown>

vfwnmsac.vv v8, v20, v4, v0.t
# CHECK-INST: vfwnmsac.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xfc]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: fc4a1457 <unknown>

vfwnmsac.vv v8, v20, v4
# CHECK-INST: vfwnmsac.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xfe]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: fe4a1457 <unknown>

vfwnmsac.vf v8, fa0, v4, v0.t
# CHECK-INST: vfwnmsac.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: fc455457 <unknown>

vfwnmsac.vf v8, fa0, v4
# CHECK-INST: vfwnmsac.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: fe455457 <unknown>
