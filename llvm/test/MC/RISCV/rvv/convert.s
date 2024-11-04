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

vfcvt.xu.f.v v8, v4, v0.t
# CHECK-INST: vfcvt.xu.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x40,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 40 48 <unknown>

vfcvt.xu.f.v v8, v4
# CHECK-INST: vfcvt.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x40,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 40 4a <unknown>

vfcvt.x.f.v v8, v4, v0.t
# CHECK-INST: vfcvt.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x40,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 40 48 <unknown>

vfcvt.x.f.v v8, v4
# CHECK-INST: vfcvt.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x40,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 40 4a <unknown>

vfcvt.f.xu.v v8, v4, v0.t
# CHECK-INST: vfcvt.f.xu.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x41,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 41 48 <unknown>

vfcvt.f.xu.v v8, v4
# CHECK-INST: vfcvt.f.xu.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x41,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 41 4a <unknown>

vfcvt.f.x.v v8, v4, v0.t
# CHECK-INST: vfcvt.f.x.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x41,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 41 48 <unknown>

vfcvt.f.x.v v8, v4
# CHECK-INST: vfcvt.f.x.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x41,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 41 4a <unknown>

vfcvt.rtz.xu.f.v v8, v4, v0.t
# CHECK-INST: vfcvt.rtz.xu.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x43,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 43 48 <unknown>

vfcvt.rtz.xu.f.v v8, v4
# CHECK-INST: vfcvt.rtz.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x43,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 43 4a <unknown>

vfcvt.rtz.x.f.v v8, v4, v0.t
# CHECK-INST: vfcvt.rtz.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x43,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 43 48 <unknown>

vfcvt.rtz.x.f.v v8, v4
# CHECK-INST: vfcvt.rtz.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x43,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 43 4a <unknown>

vfwcvt.xu.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.xu.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x44,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 44 48 <unknown>

vfwcvt.xu.f.v v8, v4
# CHECK-INST: vfwcvt.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x44,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 44 4a <unknown>

vfwcvt.x.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x44,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 44 48 <unknown>

vfwcvt.x.f.v v8, v4
# CHECK-INST: vfwcvt.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x44,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 44 4a <unknown>

vfwcvt.f.xu.v v8, v4, v0.t
# CHECK-INST: vfwcvt.f.xu.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x45,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 45 48 <unknown>

vfwcvt.f.xu.v v8, v4
# CHECK-INST: vfwcvt.f.xu.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x45,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 45 4a <unknown>

vfwcvt.f.x.v v8, v4, v0.t
# CHECK-INST: vfwcvt.f.x.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x45,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 45 48 <unknown>

vfwcvt.f.x.v v8, v4
# CHECK-INST: vfwcvt.f.x.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x45,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 45 4a <unknown>

vfwcvt.f.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.f.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x46,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 46 48 <unknown>

vfwcvt.f.f.v v8, v4
# CHECK-INST: vfwcvt.f.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x46,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 46 4a <unknown>

vfwcvt.rtz.xu.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.rtz.xu.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x47,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 47 48 <unknown>

vfwcvt.rtz.xu.f.v v8, v4
# CHECK-INST: vfwcvt.rtz.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x47,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 47 4a <unknown>

vfwcvt.rtz.x.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.rtz.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x47,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 47 48 <unknown>

vfwcvt.rtz.x.f.v v8, v4
# CHECK-INST: vfwcvt.rtz.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x47,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 47 4a <unknown>

vfncvt.xu.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.xu.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x48,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 48 48 <unknown>

vfncvt.xu.f.w v4, v4, v0.t
# CHECK-INST: vfncvt.xu.f.w v4, v4, v0.t
# CHECK-ENCODING: [0x57,0x12,0x48,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 12 48 48 <unknown>

vfncvt.xu.f.w v8, v4
# CHECK-INST: vfncvt.xu.f.w v8, v4
# CHECK-ENCODING: [0x57,0x14,0x48,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 48 4a <unknown>

vfncvt.x.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.x.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x48,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 48 48 <unknown>

vfncvt.x.f.w v8, v4
# CHECK-INST: vfncvt.x.f.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x48,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 48 4a <unknown>

vfncvt.f.xu.w v8, v4, v0.t
# CHECK-INST: vfncvt.f.xu.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x49,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 49 48 <unknown>

vfncvt.f.xu.w v8, v4
# CHECK-INST: vfncvt.f.xu.w v8, v4
# CHECK-ENCODING: [0x57,0x14,0x49,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 49 4a <unknown>

vfncvt.f.x.w v8, v4, v0.t
# CHECK-INST: vfncvt.f.x.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x49,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 49 48 <unknown>

vfncvt.f.x.w v8, v4
# CHECK-INST: vfncvt.f.x.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x49,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 49 4a <unknown>

vfncvt.f.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.f.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 4a 48 <unknown>

vfncvt.f.f.w v8, v4
# CHECK-INST: vfncvt.f.f.w v8, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 4a 4a <unknown>

vfncvt.rod.f.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.rod.f.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4a,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 4a 48 <unknown>

vfncvt.rod.f.f.w v8, v4
# CHECK-INST: vfncvt.rod.f.f.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x4a,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 4a 4a <unknown>

vfncvt.rtz.xu.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.rtz.xu.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4b,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 4b 48 <unknown>

vfncvt.rtz.xu.f.w v8, v4
# CHECK-INST: vfncvt.rtz.xu.f.w v8, v4
# CHECK-ENCODING: [0x57,0x14,0x4b,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 14 4b 4a <unknown>

vfncvt.rtz.x.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.rtz.x.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4b,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 4b 48 <unknown>

vfncvt.rtz.x.f.w v8, v4
# CHECK-INST: vfncvt.rtz.x.f.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x4b,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 94 4b 4a <unknown>
