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

vfmul.vv v8, v4, v20, v0.t
# CHECK-INST: vfmul.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x90]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 904a1457 <unknown>

vfmul.vv v8, v4, v20
# CHECK-INST: vfmul.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x92]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 924a1457 <unknown>

vfmul.vf v8, v4, fa0, v0.t
# CHECK-INST: vfmul.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x90]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 90455457 <unknown>

vfmul.vf v8, v4, fa0
# CHECK-INST: vfmul.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x92]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 92455457 <unknown>

vfwmul.vv v8, v4, v20, v0.t
# CHECK-INST: vfwmul.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xe0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e04a1457 <unknown>

vfwmul.vv v8, v4, v20
# CHECK-INST: vfwmul.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xe2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e24a1457 <unknown>

vfwmul.vf v8, v4, fa0, v0.t
# CHECK-INST: vfwmul.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0xe0]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e0455457 <unknown>

vfwmul.vf v8, v4, fa0
# CHECK-INST: vfwmul.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0xe2]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e2455457 <unknown>
