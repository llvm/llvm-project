# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+zve32f %s --M no-aliases \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:     | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+zve32f %s \
# RUN:     | llvm-objdump -d --mattr=+zve32f -M no-aliases - \
# RUN:     | FileCheck %s --check-prefix=CHECK-INST

vfredosum.vs v8, v4, v20, v0.t
# CHECK-INST: vfredosum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x0c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfredosum.vs v8, v4, v20
# CHECK-INST: vfredosum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x0e]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfredusum.vs v8, v4, v20, v0.t
# CHECK-INST: vfredusum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x04]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfredusum.vs v8, v4, v20
# CHECK-INST: vfredusum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x06]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfredmax.vs v8, v4, v20, v0.t
# CHECK-INST: vfredmax.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x1c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfredmax.vs v8, v4, v20
# CHECK-INST: vfredmax.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x1e]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfredmin.vs v8, v4, v20, v0.t
# CHECK-INST: vfredmin.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x14]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfredmin.vs v8, v4, v20
# CHECK-INST: vfredmin.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x16]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwredosum.vs v8, v4, v20, v0.t
# CHECK-INST: vfwredosum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xcc]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwredosum.vs v8, v4, v20
# CHECK-INST: vfwredosum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xce]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwredusum.vs v8, v4, v20, v0.t
# CHECK-INST: vfwredusum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xc4]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwredusum.vs v8, v4, v20
# CHECK-INST: vfwredusum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xc6]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfredosum.vs v0, v4, v20, v0.t
# CHECK-INST: vfredosum.vs v0, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x10,0x4a,0x0c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
