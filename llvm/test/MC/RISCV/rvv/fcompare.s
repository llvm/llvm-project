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

vmfeq.vv v8, v4, v20, v0.t
# CHECK-INST: vmfeq.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x60]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 604a1457 <unknown>

vmfeq.vv v8, v4, v20
# CHECK-INST: vmfeq.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x62]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 624a1457 <unknown>

vmfeq.vf v8, v4, fa0, v0.t
# CHECK-INST: vmfeq.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x60]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 60455457 <unknown>

vmfeq.vf v8, v4, fa0
# CHECK-INST: vmfeq.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x62]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 62455457 <unknown>

vmfne.vv v8, v4, v20, v0.t
# CHECK-INST: vmfne.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x70]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 704a1457 <unknown>

vmfne.vv v8, v4, v20
# CHECK-INST: vmfne.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x72]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 724a1457 <unknown>

vmfne.vf v8, v4, fa0, v0.t
# CHECK-INST: vmfne.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x70]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 70455457 <unknown>

vmfne.vf v8, v4, fa0
# CHECK-INST: vmfne.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x72]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 72455457 <unknown>

vmflt.vv v8, v4, v20, v0.t
# CHECK-INST: vmflt.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x6c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c4a1457 <unknown>

vmflt.vv v8, v4, v20
# CHECK-INST: vmflt.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x6e]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e4a1457 <unknown>

vmflt.vf v8, v4, fa0, v0.t
# CHECK-INST: vmflt.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c455457 <unknown>

vmflt.vf v8, v4, fa0
# CHECK-INST: vmflt.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e455457 <unknown>

vmfle.vv v8, v4, v20, v0.t
# CHECK-INST: vmfle.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x64]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 644a1457 <unknown>

vmfle.vv v8, v4, v20
# CHECK-INST: vmfle.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x66]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 664a1457 <unknown>

vmfle.vf v8, v4, fa0, v0.t
# CHECK-INST: vmfle.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 64455457 <unknown>

vmfle.vf v8, v4, fa0
# CHECK-INST: vmfle.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 66455457 <unknown>

vmfgt.vf v8, v4, fa0, v0.t
# CHECK-INST: vmfgt.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x74]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 74455457 <unknown>

vmfgt.vf v8, v4, fa0
# CHECK-INST: vmfgt.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x76]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 76455457 <unknown>

vmfge.vf v8, v4, fa0, v0.t
# CHECK-INST: vmfge.vf v8, v4, fa0, v0.t
# CHECK-ENCODING: [0x57,0x54,0x45,0x7c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7c455457 <unknown>

vmfge.vf v8, v4, fa0
# CHECK-INST: vmfge.vf v8, v4, fa0
# CHECK-ENCODING: [0x57,0x54,0x45,0x7e]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 7e455457 <unknown>

vmfgt.vv v8, v20, v4, v0.t
# CHECK-INST: vmflt.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x6c]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c4a1457 <unknown>

vmfgt.vv v8, v20, v4
# CHECK-INST: vmflt.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x6e]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e4a1457 <unknown>

vmfge.vv v8, v20, v4, v0.t
# CHECK-INST: vmfle.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x64]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 644a1457 <unknown>

vmfge.vv v8, v20, v4
# CHECK-INST: vmfle.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x4a,0x66]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 664a1457 <unknown>

vmfeq.vv v0, v4, v20, v0.t
# CHECK-INST: vmfeq.vv v0, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x10,0x4a,0x60]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 604a1057 <unknown>
