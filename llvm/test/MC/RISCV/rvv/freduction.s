# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+zve32f %s --M no-aliases \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+zve32f %s \
# RUN:     | llvm-objdump -d --mattr=+zve32f -M no-aliases - \
# RUN:     | FileCheck %s --check-prefix=CHECK-INST

vfredosum.vs v8, v4, v20, v0.t
# CHECK-INST: vfredosum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x0c]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfredosum.vs v8, v4, v20
# CHECK-INST: vfredosum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x0e]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfredusum.vs v8, v4, v20, v0.t
# CHECK-INST: vfredusum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x04]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfredusum.vs v8, v4, v20
# CHECK-INST: vfredusum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x06]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfredmax.vs v8, v4, v20, v0.t
# CHECK-INST: vfredmax.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x1c]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfredmax.vs v8, v4, v20
# CHECK-INST: vfredmax.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x1e]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfredmin.vs v8, v4, v20, v0.t
# CHECK-INST: vfredmin.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x14]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfredmin.vs v8, v4, v20
# CHECK-INST: vfredmin.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x16]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfwredosum.vs v8, v4, v20, v0.t
# CHECK-INST: vfwredosum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xcc]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfwredosum.vs v8, v4, v20
# CHECK-INST: vfwredosum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xce]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfwredusum.vs v8, v4, v20, v0.t
# CHECK-INST: vfwredusum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xc4]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfwredusum.vs v8, v4, v20
# CHECK-INST: vfwredusum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xc6]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}

vfredosum.vs v0, v4, v20, v0.t
# CHECK-INST: vfredosum.vs v0, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x10,0x4a,0x0c]
# expected-error-re@-3 {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
