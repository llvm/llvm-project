# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:         --mattr=+f \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:         --mattr=+f \
# RUN:        | llvm-objdump -d --mattr=+v --mattr=+f - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vfcvt.xu.f.v v8, v4, v0.t
# CHECK-INST: vfcvt.xu.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x40,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfcvt.xu.f.v v8, v4
# CHECK-INST: vfcvt.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x40,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfcvt.x.f.v v8, v4, v0.t
# CHECK-INST: vfcvt.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x40,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfcvt.x.f.v v8, v4
# CHECK-INST: vfcvt.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x40,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfcvt.f.xu.v v8, v4, v0.t
# CHECK-INST: vfcvt.f.xu.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x41,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfcvt.f.xu.v v8, v4
# CHECK-INST: vfcvt.f.xu.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x41,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfcvt.f.x.v v8, v4, v0.t
# CHECK-INST: vfcvt.f.x.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x41,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfcvt.f.x.v v8, v4
# CHECK-INST: vfcvt.f.x.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x41,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfcvt.rtz.xu.f.v v8, v4, v0.t
# CHECK-INST: vfcvt.rtz.xu.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x43,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfcvt.rtz.xu.f.v v8, v4
# CHECK-INST: vfcvt.rtz.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x43,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfcvt.rtz.x.f.v v8, v4, v0.t
# CHECK-INST: vfcvt.rtz.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x43,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfcvt.rtz.x.f.v v8, v4
# CHECK-INST: vfcvt.rtz.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x43,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwcvt.xu.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.xu.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x44,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwcvt.xu.f.v v8, v4
# CHECK-INST: vfwcvt.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x44,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwcvt.x.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x44,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwcvt.x.f.v v8, v4
# CHECK-INST: vfwcvt.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x44,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwcvt.f.xu.v v8, v4, v0.t
# CHECK-INST: vfwcvt.f.xu.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x45,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwcvt.f.xu.v v8, v4
# CHECK-INST: vfwcvt.f.xu.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x45,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwcvt.f.x.v v8, v4, v0.t
# CHECK-INST: vfwcvt.f.x.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x45,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwcvt.f.x.v v8, v4
# CHECK-INST: vfwcvt.f.x.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x45,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwcvt.f.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.f.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x46,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwcvt.f.f.v v8, v4
# CHECK-INST: vfwcvt.f.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x46,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwcvt.rtz.xu.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.rtz.xu.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x47,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwcvt.rtz.xu.f.v v8, v4
# CHECK-INST: vfwcvt.rtz.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x47,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwcvt.rtz.x.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.rtz.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x47,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfwcvt.rtz.x.f.v v8, v4
# CHECK-INST: vfwcvt.rtz.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x47,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.xu.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.xu.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x48,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.xu.f.w v4, v4, v0.t
# CHECK-INST: vfncvt.xu.f.w v4, v4, v0.t
# CHECK-ENCODING: [0x57,0x12,0x48,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.xu.f.w v8, v4
# CHECK-INST: vfncvt.xu.f.w v8, v4
# CHECK-ENCODING: [0x57,0x14,0x48,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.x.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.x.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x48,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.x.f.w v8, v4
# CHECK-INST: vfncvt.x.f.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x48,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.f.xu.w v8, v4, v0.t
# CHECK-INST: vfncvt.f.xu.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x49,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.f.xu.w v8, v4
# CHECK-INST: vfncvt.f.xu.w v8, v4
# CHECK-ENCODING: [0x57,0x14,0x49,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.f.x.w v8, v4, v0.t
# CHECK-INST: vfncvt.f.x.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x49,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.f.x.w v8, v4
# CHECK-INST: vfncvt.f.x.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x49,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.f.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.f.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.f.f.w v8, v4
# CHECK-INST: vfncvt.f.f.w v8, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.rod.f.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.rod.f.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4a,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.rod.f.f.w v8, v4
# CHECK-INST: vfncvt.rod.f.f.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x4a,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.rtz.xu.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.rtz.xu.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4b,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.rtz.xu.f.w v8, v4
# CHECK-INST: vfncvt.rtz.xu.f.w v8, v4
# CHECK-ENCODING: [0x57,0x14,0x4b,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.rtz.x.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.rtz.x.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4b,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}

vfncvt.rtz.x.f.w v8, v4
# CHECK-INST: vfncvt.rtz.x.f.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x4b,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
