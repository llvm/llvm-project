# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+zve32f %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:     | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+zve32f %s \
# RUN:     | llvm-objdump -d --mattr=+zve32f - \
# RUN:     | FileCheck %s --check-prefix=CHECK-INST

vfmacc.vv v8, v20, v4, v0.t
# CHECK-INST: vfmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xb0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfmacc.vv v8, v20, v4
# CHECK-INST: vfmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xb2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfmacc.vf v8, fa0, v4, v0.t
# CHECK-INST: vfmacc.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xb0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfmacc.vf v8, fa0, v4
# CHECK-INST: vfmacc.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xb2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfnmacc.vv v8, v20, v4, v0.t
# CHECK-INST: vfnmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xb4]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfnmacc.vv v8, v20, v4
# CHECK-INST: vfnmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xb6]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfnmacc.vf v8, fa0, v4, v0.t
# CHECK-INST: vfnmacc.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xb4]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfnmacc.vf v8, fa0, v4
# CHECK-INST: vfnmacc.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xb6]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfmsac.vv v8, v20, v4, v0.t
# CHECK-INST: vfmsac.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xb8]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfmsac.vv v8, v20, v4
# CHECK-INST: vfmsac.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xba]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfmsac.vf v8, fa0, v4, v0.t
# CHECK-INST: vfmsac.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xb8]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfmsac.vf v8, fa0, v4
# CHECK-INST: vfmsac.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xba]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfnmsac.vv v8, v20, v4, v0.t
# CHECK-INST: vfnmsac.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xbc]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfnmsac.vv v8, v20, v4
# CHECK-INST: vfnmsac.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xbe]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfnmsac.vf v8, fa0, v4, v0.t
# CHECK-INST: vfnmsac.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xbc]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfnmsac.vf v8, fa0, v4
# CHECK-INST: vfnmsac.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xbe]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfmadd.vv v8, v20, v4, v0.t
# CHECK-INST: vfmadd.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xa0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfmadd.vv v8, v20, v4
# CHECK-INST: vfmadd.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xa2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfmadd.vf v8, fa0, v4, v0.t
# CHECK-INST: vfmadd.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xa0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfmadd.vf v8, fa0, v4
# CHECK-INST: vfmadd.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xa2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfnmadd.vv v8, v20, v4, v0.t
# CHECK-INST: vfnmadd.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xa4]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfnmadd.vv v8, v20, v4
# CHECK-INST: vfnmadd.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xa6]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfnmadd.vf v8, fa0, v4, v0.t
# CHECK-INST: vfnmadd.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfnmadd.vf v8, fa0, v4
# CHECK-INST: vfnmadd.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfmsub.vv v8, v20, v4, v0.t
# CHECK-INST: vfmsub.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xa8]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfmsub.vv v8, v20, v4
# CHECK-INST: vfmsub.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xaa]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfmsub.vf v8, fa0, v4, v0.t
# CHECK-INST: vfmsub.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xa8]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfmsub.vf v8, fa0, v4
# CHECK-INST: vfmsub.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xaa]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfnmsub.vv v8, v20, v4, v0.t
# CHECK-INST: vfnmsub.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xac]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfnmsub.vv v8, v20, v4
# CHECK-INST: vfnmsub.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xae]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfnmsub.vf v8, fa0, v4, v0.t
# CHECK-INST: vfnmsub.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfnmsub.vf v8, fa0, v4
# CHECK-INST: vfnmsub.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwmacc.vv v8, v20, v4, v0.t
# CHECK-INST: vfwmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xf0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwmacc.vv v8, v20, v4
# CHECK-INST: vfwmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xf2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwmacc.vf v8, fa0, v4, v0.t
# CHECK-INST: vfwmacc.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xf0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwmacc.vf v8, fa0, v4
# CHECK-INST: vfwmacc.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xf2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwnmacc.vv v8, v20, v4, v0.t
# CHECK-INST: vfwnmacc.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xf4]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwnmacc.vv v8, v20, v4
# CHECK-INST: vfwnmacc.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xf6]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwnmacc.vf v8, fa0, v4, v0.t
# CHECK-INST: vfwnmacc.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xf4]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwnmacc.vf v8, fa0, v4
# CHECK-INST: vfwnmacc.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xf6]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwmsac.vv v8, v20, v4, v0.t
# CHECK-INST: vfwmsac.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xf8]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwmsac.vv v8, v20, v4
# CHECK-INST: vfwmsac.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xfa]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwmsac.vf v8, fa0, v4, v0.t
# CHECK-INST: vfwmsac.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xf8]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwmsac.vf v8, fa0, v4
# CHECK-INST: vfwmsac.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xfa]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwnmsac.vv v8, v20, v4, v0.t
# CHECK-INST: vfwnmsac.vv v8, v20, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xfc]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwnmsac.vv v8, v20, v4
# CHECK-INST: vfwnmsac.vv v8, v20, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0xfe]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwnmsac.vf v8, fa0, v4, v0.t
# CHECK-INST: vfwnmsac.vf v8, fa0, v4, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xfc]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwnmsac.vf v8, fa0, v4
# CHECK-INST: vfwnmsac.vf v8, fa0, v4
# CHECK-ENCODING: [0x57,0x54,0x45,0xfe]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
