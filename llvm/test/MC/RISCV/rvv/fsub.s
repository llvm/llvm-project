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

vfsub.vv v8, v4, v20, v0.t
# CHECK-INST: vfsub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x08]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 084a1457 <unknown>

vfsub.vv v8, v4, v20
# CHECK-INST: vfsub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x0a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0a4a1457 <unknown>

vfsub.vf v8, v4, fa0, v0.t
# CHECK-INST: vfsub.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x08]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 08455457 <unknown>

vfsub.vf v8, v4, fa0
# CHECK-INST: vfsub.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x0a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0a455457 <unknown>

vfrsub.vf v8, v4, fa0, v0.t
# CHECK-INST: vfrsub.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x9c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 9c455457 <unknown>

vfrsub.vf v8, v4, fa0
# CHECK-INST: vfrsub.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x9e]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 9e455457 <unknown>

vfwsub.vv v8, v4, v20, v0.t
# CHECK-INST: vfwsub.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xc8]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c84a1457 <unknown>

vfwsub.vv v8, v4, v20
# CHECK-INST: vfwsub.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xca]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ca4a1457 <unknown>

vfwsub.vf v8, v4, fa0, v0.t
# CHECK-INST: vfwsub.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xc8]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c8455457 <unknown>

vfwsub.vf v8, v4, fa0
# CHECK-INST: vfwsub.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0xca]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ca455457 <unknown>

vfwsub.wv v8, v4, v20, v0.t
# CHECK-INST: vfwsub.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xd8]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: d84a1457 <unknown>

vfwsub.wv v8, v4, v20
# CHECK-INST: vfwsub.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xda]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: da4a1457 <unknown>

vfwsub.wf v8, v4, fa0, v0.t
# CHECK-INST: vfwsub.wf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xd8]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: d8455457 <unknown>

vfwsub.wf v8, v4, fa0
# CHECK-INST: vfwsub.wf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0xda]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: da455457 <unknown>
