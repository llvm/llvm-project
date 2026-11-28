# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+zve32f %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+zve32f %s \
# RUN:     | llvm-objdump -d --mattr=+zve32f - \
# RUN:     | FileCheck %s --check-prefix=CHECK-INST

vmfeq.vv v8, v4, v20, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfeq.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x60]

vmfeq.vv v8, v4, v20 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfeq.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x62]

vmfeq.vf v8, v4, fa0, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfeq.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x60]

vmfeq.vf v8, v4, fa0 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfeq.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x62]

vmfne.vv v8, v4, v20, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfne.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x70]

vmfne.vv v8, v4, v20 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfne.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x72]

vmfne.vf v8, v4, fa0, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfne.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x70]

vmfne.vf v8, v4, fa0 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfne.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x72]

vmflt.vv v8, v4, v20, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmflt.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x6c]

vmflt.vv v8, v4, v20 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmflt.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x6e]

vmflt.vf v8, v4, fa0, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmflt.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x6c]

vmflt.vf v8, v4, fa0 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmflt.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x6e]

vmfle.vv v8, v4, v20, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfle.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x64]

vmfle.vv v8, v4, v20 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfle.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x66]

vmfle.vf v8, v4, fa0, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfle.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x64]

vmfle.vf v8, v4, fa0 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfle.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x66]

vmfgt.vf v8, v4, fa0, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfgt.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x74]

vmfgt.vf v8, v4, fa0 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfgt.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x76]

vmfge.vf v8, v4, fa0, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfge.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x7c]

vmfge.vf v8, v4, fa0 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfge.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x7e]

vmfgt.vv v8, v20, v4, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmflt.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x6c]

vmfgt.vv v8, v20, v4 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmflt.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x6e]

vmfge.vv v8, v20, v4, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfle.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x64]

vmfge.vv v8, v20, v4 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfle.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x66]

vmfeq.vv v0, v4, v20, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vmfeq.vv v0, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x10,0x4a,0x60]
