# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:         --mattr=+f --riscv-no-aliases \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:         --mattr=+f \
# RUN:        | llvm-objdump -d --mattr=+v --mattr=+f \
# RUN:        -M no-aliases - | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:         --mattr=+f \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vfredosum.vs v8, v4, v20, v0.t
# CHECK-INST: vfredosum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x0c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0c4a1457 <unknown>

vfredosum.vs v8, v4, v20
# CHECK-INST: vfredosum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x0e]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0e4a1457 <unknown>

vfredusum.vs v8, v4, v20, v0.t
# CHECK-INST: vfredusum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x04]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 044a1457 <unknown>

vfredusum.vs v8, v4, v20
# CHECK-INST: vfredusum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x06]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 064a1457 <unknown>

vfredmax.vs v8, v4, v20, v0.t
# CHECK-INST: vfredmax.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x1c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 1c4a1457 <unknown>

vfredmax.vs v8, v4, v20
# CHECK-INST: vfredmax.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x1e]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 1e4a1457 <unknown>

vfredmin.vs v8, v4, v20, v0.t
# CHECK-INST: vfredmin.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x14]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 144a1457 <unknown>

vfredmin.vs v8, v4, v20
# CHECK-INST: vfredmin.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x16]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 164a1457 <unknown>

vfwredosum.vs v8, v4, v20, v0.t
# CHECK-INST: vfwredosum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xcc]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cc4a1457 <unknown>

vfwredosum.vs v8, v4, v20
# CHECK-INST: vfwredosum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xce]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ce4a1457 <unknown>

vfwredusum.vs v8, v4, v20, v0.t
# CHECK-INST: vfwredusum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0xc4]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c44a1457 <unknown>

vfwredusum.vs v8, v4, v20
# CHECK-INST: vfwredusum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0xc6]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c64a1457 <unknown>

vfredosum.vs v0, v4, v20, v0.t
# CHECK-INST: vfredosum.vs v0, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x10,0x4a,0x0c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0c4a1057 <unknown>
