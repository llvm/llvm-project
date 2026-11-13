# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+zve32f %s --M no-aliases \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+zve32f %s \
# RUN:     | llvm-objdump -d --mattr=+zve32f -M no-aliases - \
# RUN:     | FileCheck %s --check-prefix=CHECK-INST

vfredosum.vs v8, v4, v20, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfredosum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x0c]

vfredosum.vs v8, v4, v20 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfredosum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x0e]

vfredusum.vs v8, v4, v20, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfredusum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x04]

vfredusum.vs v8, v4, v20 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfredusum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x06]

vfredmax.vs v8, v4, v20, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfredmax.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x1c]

vfredmax.vs v8, v4, v20 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfredmax.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x1e]

vfredmin.vs v8, v4, v20, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfredmin.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x14]

vfredmin.vs v8, v4, v20 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfredmin.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x16]

vfwredosum.vs v8, v4, v20, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfwredosum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xcc]

vfwredosum.vs v8, v4, v20 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfwredosum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xce]

vfwredusum.vs v8, v4, v20, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfwredusum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xc4]

vfwredusum.vs v8, v4, v20 # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfwredusum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xc6]

vfredosum.vs v0, v4, v20, v0.t # expected-error-re {{instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vfredosum.vs v0, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x10,0x4a,0x0c]
