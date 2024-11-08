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
# CHECK-UNKNOWN: 48401457 <unknown>

vfcvt.xu.f.v v8, v4
# CHECK-INST: vfcvt.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x40,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a401457 <unknown>

vfcvt.x.f.v v8, v4, v0.t
# CHECK-INST: vfcvt.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x40,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48409457 <unknown>

vfcvt.x.f.v v8, v4
# CHECK-INST: vfcvt.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x40,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a409457 <unknown>

vfcvt.f.xu.v v8, v4, v0.t
# CHECK-INST: vfcvt.f.xu.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x41,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48411457 <unknown>

vfcvt.f.xu.v v8, v4
# CHECK-INST: vfcvt.f.xu.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x41,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a411457 <unknown>

vfcvt.f.x.v v8, v4, v0.t
# CHECK-INST: vfcvt.f.x.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x41,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48419457 <unknown>

vfcvt.f.x.v v8, v4
# CHECK-INST: vfcvt.f.x.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x41,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a419457 <unknown>

vfcvt.rtz.xu.f.v v8, v4, v0.t
# CHECK-INST: vfcvt.rtz.xu.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x43,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48431457 <unknown>

vfcvt.rtz.xu.f.v v8, v4
# CHECK-INST: vfcvt.rtz.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x43,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a431457 <unknown>

vfcvt.rtz.x.f.v v8, v4, v0.t
# CHECK-INST: vfcvt.rtz.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x43,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48439457 <unknown>

vfcvt.rtz.x.f.v v8, v4
# CHECK-INST: vfcvt.rtz.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x43,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a439457 <unknown>

vfwcvt.xu.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.xu.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x44,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48441457 <unknown>

vfwcvt.xu.f.v v8, v4
# CHECK-INST: vfwcvt.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x44,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a441457 <unknown>

vfwcvt.x.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x44,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48449457 <unknown>

vfwcvt.x.f.v v8, v4
# CHECK-INST: vfwcvt.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x44,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a449457 <unknown>

vfwcvt.f.xu.v v8, v4, v0.t
# CHECK-INST: vfwcvt.f.xu.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x45,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48451457 <unknown>

vfwcvt.f.xu.v v8, v4
# CHECK-INST: vfwcvt.f.xu.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x45,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a451457 <unknown>

vfwcvt.f.x.v v8, v4, v0.t
# CHECK-INST: vfwcvt.f.x.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x45,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48459457 <unknown>

vfwcvt.f.x.v v8, v4
# CHECK-INST: vfwcvt.f.x.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x45,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a459457 <unknown>

vfwcvt.f.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.f.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x46,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48461457 <unknown>

vfwcvt.f.f.v v8, v4
# CHECK-INST: vfwcvt.f.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x46,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a461457 <unknown>

vfwcvt.rtz.xu.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.rtz.xu.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x47,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48471457 <unknown>

vfwcvt.rtz.xu.f.v v8, v4
# CHECK-INST: vfwcvt.rtz.xu.f.v v8, v4
# CHECK-ENCODING: [0x57,0x14,0x47,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a471457 <unknown>

vfwcvt.rtz.x.f.v v8, v4, v0.t
# CHECK-INST: vfwcvt.rtz.x.f.v v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x47,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48479457 <unknown>

vfwcvt.rtz.x.f.v v8, v4
# CHECK-INST: vfwcvt.rtz.x.f.v v8, v4
# CHECK-ENCODING: [0x57,0x94,0x47,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a479457 <unknown>

vfncvt.xu.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.xu.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x48,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48481457 <unknown>

vfncvt.xu.f.w v4, v4, v0.t
# CHECK-INST: vfncvt.xu.f.w v4, v4, v0.t
# CHECK-ENCODING: [0x57,0x12,0x48,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48481257 <unknown>

vfncvt.xu.f.w v8, v4
# CHECK-INST: vfncvt.xu.f.w v8, v4
# CHECK-ENCODING: [0x57,0x14,0x48,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a481457 <unknown>

vfncvt.x.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.x.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x48,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48489457 <unknown>

vfncvt.x.f.w v8, v4
# CHECK-INST: vfncvt.x.f.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x48,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a489457 <unknown>

vfncvt.f.xu.w v8, v4, v0.t
# CHECK-INST: vfncvt.f.xu.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x49,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48491457 <unknown>

vfncvt.f.xu.w v8, v4
# CHECK-INST: vfncvt.f.xu.w v8, v4
# CHECK-ENCODING: [0x57,0x14,0x49,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a491457 <unknown>

vfncvt.f.x.w v8, v4, v0.t
# CHECK-INST: vfncvt.f.x.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x49,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48499457 <unknown>

vfncvt.f.x.w v8, v4
# CHECK-INST: vfncvt.f.x.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x49,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a499457 <unknown>

vfncvt.f.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.f.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4a,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 484a1457 <unknown>

vfncvt.f.f.w v8, v4
# CHECK-INST: vfncvt.f.f.w v8, v4
# CHECK-ENCODING: [0x57,0x14,0x4a,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a4a1457 <unknown>

vfncvt.rod.f.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.rod.f.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4a,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 484a9457 <unknown>

vfncvt.rod.f.f.w v8, v4
# CHECK-INST: vfncvt.rod.f.f.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x4a,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a4a9457 <unknown>

vfncvt.rtz.xu.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.rtz.xu.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x14,0x4b,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 484b1457 <unknown>

vfncvt.rtz.xu.f.w v8, v4
# CHECK-INST: vfncvt.rtz.xu.f.w v8, v4
# CHECK-ENCODING: [0x57,0x14,0x4b,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a4b1457 <unknown>

vfncvt.rtz.x.f.w v8, v4, v0.t
# CHECK-INST: vfncvt.rtz.x.f.w v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x94,0x4b,0x48]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 484b9457 <unknown>

vfncvt.rtz.x.f.w v8, v4
# CHECK-INST: vfncvt.rtz.x.f.w v8, v4
# CHECK-ENCODING: [0x57,0x94,0x4b,0x4a]
# CHECK-ERROR: instruction requires the following: 'V'{{.*}}'Zve32f' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a4b9457 <unknown>
