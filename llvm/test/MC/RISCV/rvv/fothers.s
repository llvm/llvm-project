# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+zve32f %s --M no-aliases \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+zve32f %s \
# RUN:   | llvm-objdump -d --mattr=+zve32f -M no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST

vfsqrt.v v8, v4, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfsqrt.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x40,0x4c]

vfsqrt.v v8, v4 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfsqrt.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x40,0x4e]

vfrsqrt7.v v8, v4, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfrsqrt7.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x42,0x4c]

vfrsqrt7.v v8, v4 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfrsqrt7.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x42,0x4e]

vfrec7.v v8, v4, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfrec7.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x42,0x4c]

vfrec7.v v8, v4 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfrec7.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x42,0x4e]

vfclass.v v8, v4, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfclass.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x48,0x4c]

vfclass.v v8, v4 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfclass.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x48,0x4e]

vfmerge.vfm v8, v4, fa0, v0 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfmerge.vfm v8, v4, fa0, v0
# CHECK-ENCODING: [0x57,0x54,0x45,0x5c]

vfslide1up.vf v8, v4, fa0, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfslide1up.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x38]

vfslide1up.vf v8, v4, fa0 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfslide1up.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x3a]

vfslide1down.vf v8, v4, fa0, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfslide1down.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x3c]

vfslide1down.vf v8, v4, fa0 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfslide1down.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x3e]
