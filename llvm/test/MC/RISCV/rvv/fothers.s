# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:   --mattr=+f --M no-aliases \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:   --mattr=+f | llvm-objdump -d --mattr=+v --mattr=+f -M no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:   --mattr=+f | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vfsqrt.v v8, v4, v0.t
# CHECK-INST: vfsqrt.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x40,0x4c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4c401457 <unknown>

vfsqrt.v v8, v4
# CHECK-INST: vfsqrt.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x40,0x4e]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4e401457 <unknown>

vfrsqrt7.v v8, v4, v0.t
# CHECK-INST: vfrsqrt7.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x42,0x4c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4c421457 <unknown>

vfrsqrt7.v v8, v4
# CHECK-INST: vfrsqrt7.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x42,0x4e]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4e421457 <unknown>

vfrec7.v v8, v4, v0.t
# CHECK-INST: vfrec7.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x42,0x4c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4c429457 <unknown>

vfrec7.v v8, v4
# CHECK-INST: vfrec7.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x42,0x4e]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4e429457 <unknown>

vfclass.v v8, v4, v0.t
# CHECK-INST: vfclass.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x48,0x4c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4c481457 <unknown>

vfclass.v v8, v4
# CHECK-INST: vfclass.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x48,0x4e]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4e481457 <unknown>

vfmerge.vfm v8, v4, fa0, v0
# CHECK-INST: vfmerge.vfm v8, v4, fa0, v0
# CHECK-ENCODING: [0x57,0x54,0x45,0x5c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 5c455457 <unknown>

vfslide1up.vf v8, v4, fa0, v0.t
# CHECK-INST: vfslide1up.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x38]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 38455457 <unknown>

vfslide1up.vf v8, v4, fa0
# CHECK-INST: vfslide1up.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x3a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 3a455457 <unknown>

vfslide1down.vf v8, v4, fa0, v0.t
# CHECK-INST: vfslide1down.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x3c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 3c455457 <unknown>

vfslide1down.vf v8, v4, fa0
# CHECK-INST: vfslide1down.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x3e]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 3e455457 <unknown>
