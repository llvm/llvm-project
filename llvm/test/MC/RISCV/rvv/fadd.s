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

vfadd.vv v8, v4, v20, v0.t
# CHECK-INST: vfadd.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x00]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 004a1457 <unknown>

vfadd.vv v8, v4, v20
# CHECK-INST: vfadd.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x02]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 024a1457 <unknown>

vfadd.vf v8, v4, fa0, v0.t
# CHECK-INST: vfadd.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x00]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 00455457 <unknown>

vfadd.vf v8, v4, fa0
# CHECK-INST: vfadd.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x02]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 02455457 <unknown>

vfwadd.vv v8, v4, v20, v0.t
# CHECK-INST: vfwadd.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xc0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c04a1457 <unknown>

vfwadd.vv v8, v4, v20
# CHECK-INST: vfwadd.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xc2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c24a1457 <unknown>

vfwadd.vf v8, v4, fa0, v0.t
# CHECK-INST: vfwadd.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xc0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c0455457 <unknown>

vfwadd.vf v8, v4, fa0
# CHECK-INST: vfwadd.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0xc2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c2455457 <unknown>

vfwadd.wv v8, v4, v20, v0.t
# CHECK-INST: vfwadd.wv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xd0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: d04a1457 <unknown>

vfwadd.wv v8, v4, v20
# CHECK-INST: vfwadd.wv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xd2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: d24a1457 <unknown>

vfwadd.wf v8, v4, fa0, v0.t
# CHECK-INST: vfwadd.wf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xd0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: d0455457 <unknown>

vfwadd.wf v8, v4, fa0
# CHECK-INST: vfwadd.wf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0xd2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: d2455457 <unknown>
