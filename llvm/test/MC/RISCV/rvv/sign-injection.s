# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:         --mattr=+f \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:         --mattr=+f \
# RUN:        | llvm-objdump -d --mattr=+v --mattr=+f - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vfsgnj.vv v8, v4, v20, v0.t
# CHECK-INST: vfsgnj.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x20]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfsgnj.vv v8, v4, v20
# CHECK-INST: vfsgnj.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x22]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfsgnj.vf v8, v4, fa0, v0.t
# CHECK-INST: vfsgnj.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x20]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfsgnj.vf v8, v4, fa0
# CHECK-INST: vfsgnj.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x22]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfsgnjn.vv v8, v4, v20, v0.t
# CHECK-INST: vfsgnjn.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x24]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfsgnjn.vv v8, v4, v20
# CHECK-INST: vfsgnjn.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x26]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfsgnjn.vf v8, v4, fa0, v0.t
# CHECK-INST: vfsgnjn.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfsgnjn.vf v8, v4, fa0
# CHECK-INST: vfsgnjn.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfsgnjx.vv v8, v4, v20, v0.t
# CHECK-INST: vfsgnjx.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x28]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfsgnjx.vv v8, v4, v20
# CHECK-INST: vfsgnjx.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x2a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfsgnjx.vf v8, v4, fa0, v0.t
# CHECK-INST: vfsgnjx.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x28]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfsgnjx.vf v8, v4, fa0
# CHECK-INST: vfsgnjx.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x2a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
