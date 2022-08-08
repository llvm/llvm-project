# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:   --riscv-no-aliases | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:   | llvm-objdump -d --mattr=+v -M no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vlm.v v0, (a0)
# CHECK-INST: vlm.v v0, (a0)
# CHECK-ENCODING: [0x07,0x00,0xb5,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 00 b5 02 <unknown>

vlm.v v8, (a0)
# CHECK-INST: vlm.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0xb5,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 04 b5 02 <unknown>

vle8.v v8, (a0), v0.t
# CHECK-INST: vle8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 04 05 00 <unknown>

vle8.v v8, (a0)
# CHECK-INST: vle8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 04 05 02 <unknown>

vle16.v v8, (a0), v0.t
# CHECK-INST: vle16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 54 05 00 <unknown>

vle16.v v8, (a0)
# CHECK-INST: vle16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 54 05 02 <unknown>

vle32.v v8, (a0), v0.t
# CHECK-INST: vle32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 64 05 00 <unknown>

vle32.v v8, (a0)
# CHECK-INST: vle32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 64 05 02 <unknown>

vle64.v v8, (a0), v0.t
# CHECK-INST: vle64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 74 05 00 <unknown>

vle64.v v8, (a0)
# CHECK-INST: vle64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 74 05 02 <unknown>

vle8ff.v v8, (a0), v0.t
# CHECK-INST: vle8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 04 05 01 <unknown>

vle8ff.v v8, (a0)
# CHECK-INST: vle8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 04 05 03 <unknown>

vle16ff.v v8, (a0), v0.t
# CHECK-INST: vle16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 54 05 01 <unknown>

vle16ff.v v8, (a0)
# CHECK-INST: vle16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 54 05 03 <unknown>

vle32ff.v v8, (a0), v0.t
# CHECK-INST: vle32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 64 05 01 <unknown>

vle32ff.v v8, (a0)
# CHECK-INST: vle32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 64 05 03 <unknown>

vle64ff.v v8, (a0), v0.t
# CHECK-INST: vle64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 74 05 01 <unknown>

vle64ff.v v8, (a0)
# CHECK-INST: vle64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 74 05 03 <unknown>

vlse8.v v8, (a0), a1, v0.t
# CHECK-INST: vlse8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 04 b5 08 <unknown>

vlse8.v v8, (a0), a1
# CHECK-INST: vlse8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 04 b5 0a <unknown>

vlse16.v v8, (a0), a1, v0.t
# CHECK-INST: vlse16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 54 b5 08 <unknown>

vlse16.v v8, (a0), a1
# CHECK-INST: vlse16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 54 b5 0a <unknown>

vlse32.v v8, (a0), a1, v0.t
# CHECK-INST: vlse32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 64 b5 08 <unknown>

vlse32.v v8, (a0), a1
# CHECK-INST: vlse32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 64 b5 0a <unknown>

vlse64.v v8, (a0), a1, v0.t
# CHECK-INST: vlse64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 74 b5 08 <unknown>

vlse64.v v8, (a0), a1
# CHECK-INST: vlse64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 74 b5 0a <unknown>

vluxei8.v v8, (a0), v4, v0.t
# CHECK-INST: vluxei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x04]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 04 45 04 <unknown>

vluxei8.v v8, (a0), v4
# CHECK-INST: vluxei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x06]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 04 45 06 <unknown>

vluxei16.v v8, (a0), v4, v0.t
# CHECK-INST: vluxei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x04]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 54 45 04 <unknown>

vluxei16.v v8, (a0), v4
# CHECK-INST: vluxei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x06]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 54 45 06 <unknown>

vluxei32.v v8, (a0), v4, v0.t
# CHECK-INST: vluxei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x04]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 64 45 04 <unknown>

vluxei32.v v8, (a0), v4
# CHECK-INST: vluxei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x06]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 64 45 06 <unknown>

vluxei64.v v8, (a0), v4, v0.t
# CHECK-INST: vluxei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x04]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 74 45 04 <unknown>

vluxei64.v v8, (a0), v4
# CHECK-INST: vluxei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x06]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 74 45 06 <unknown>

vloxei8.v v8, (a0), v4, v0.t
# CHECK-INST: vloxei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 04 45 0c <unknown>

vloxei8.v v8, (a0), v4
# CHECK-INST: vloxei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 04 45 0e <unknown>

vloxei16.v v8, (a0), v4, v0.t
# CHECK-INST: vloxei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 54 45 0c <unknown>

vloxei16.v v8, (a0), v4
# CHECK-INST: vloxei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 54 45 0e <unknown>

vloxei32.v v8, (a0), v4, v0.t
# CHECK-INST: vloxei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 64 45 0c <unknown>

vloxei32.v v8, (a0), v4
# CHECK-INST: vloxei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 64 45 0e <unknown>

vloxei64.v v8, (a0), v4, v0.t
# CHECK-INST: vloxei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 74 45 0c <unknown>

vloxei64.v v8, (a0), v4
# CHECK-INST: vloxei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 74 45 0e <unknown>

vl1re8.v v8, (a0)
# CHECK-INST: vl1re8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x85,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 04 85 02 <unknown>

vl1re16.v v8, (a0)
# CHECK-INST: vl1re16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x85,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 54 85 02 <unknown>

vl1re32.v v8, (a0)
# CHECK-INST: vl1re32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x85,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 64 85 02 <unknown>

vl1re64.v v8, (a0)
# CHECK-INST: vl1re64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x85,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 74 85 02 <unknown>

vl2re8.v v8, (a0)
# CHECK-INST: vl2re8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x85,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 04 85 22 <unknown>

vl2re16.v v8, (a0)
# CHECK-INST: vl2re16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x85,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 54 85 22 <unknown>

vl2re32.v v8, (a0)
# CHECK-INST: vl2re32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x85,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 64 85 22 <unknown>

vl2re64.v v8, (a0)
# CHECK-INST: vl2re64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x85,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 74 85 22 <unknown>

vl4re8.v v8, (a0)
# CHECK-INST: vl4re8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x85,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 04 85 62 <unknown>

vl4re16.v v8, (a0)
# CHECK-INST: vl4re16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x85,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 54 85 62 <unknown>

vl4re32.v v8, (a0)
# CHECK-INST: vl4re32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x85,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 64 85 62 <unknown>

vl4re64.v v8, (a0)
# CHECK-INST: vl4re64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x85,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 74 85 62 <unknown>

vl8re8.v v8, (a0)
# CHECK-INST: vl8re8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x85,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 04 85 e2 <unknown>

vl8re16.v v8, (a0)
# CHECK-INST: vl8re16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x85,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 54 85 e2 <unknown>

vl8re32.v v8, (a0)
# CHECK-INST: vl8re32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x85,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 64 85 e2 <unknown>

vl8re64.v v8, (a0)
# CHECK-INST: vl8re64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x85,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 07 74 85 e2 <unknown>
