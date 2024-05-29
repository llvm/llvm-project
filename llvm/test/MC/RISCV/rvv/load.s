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
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 02b50007 <unknown>

vlm.v v8, (a0)
# CHECK-INST: vlm.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0xb5,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 02b50407 <unknown>

vle8.v v8, (a0), v0.t
# CHECK-INST: vle8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 00050407 <unknown>

vle8.v v8, (a0)
# CHECK-INST: vle8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 02050407 <unknown>

vle16.v v8, (a0), v0.t
# CHECK-INST: vle16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 00055407 <unknown>

vle16.v v8, (a0)
# CHECK-INST: vle16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 02055407 <unknown>

vle32.v v8, (a0), v0.t
# CHECK-INST: vle32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 00056407 <unknown>

vle32.v v8, (a0)
# CHECK-INST: vle32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 02056407 <unknown>

vle64.v v8, (a0), v0.t
# CHECK-INST: vle64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x00]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 00057407 <unknown>

vle64.v v8, (a0)
# CHECK-INST: vle64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 02057407 <unknown>

vle8ff.v v8, (a0), v0.t
# CHECK-INST: vle8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 01050407 <unknown>

vle8ff.v v8, (a0)
# CHECK-INST: vle8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 03050407 <unknown>

vle16ff.v v8, (a0), v0.t
# CHECK-INST: vle16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 01055407 <unknown>

vle16ff.v v8, (a0)
# CHECK-INST: vle16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 03055407 <unknown>

vle32ff.v v8, (a0), v0.t
# CHECK-INST: vle32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 01056407 <unknown>

vle32ff.v v8, (a0)
# CHECK-INST: vle32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 03056407 <unknown>

vle64ff.v v8, (a0), v0.t
# CHECK-INST: vle64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 01057407 <unknown>

vle64ff.v v8, (a0)
# CHECK-INST: vle64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x03]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 03057407 <unknown>

vlse8.v v8, (a0), a1, v0.t
# CHECK-INST: vlse8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 08b50407 <unknown>

vlse8.v v8, (a0), a1
# CHECK-INST: vlse8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0ab50407 <unknown>

vlse16.v v8, (a0), a1, v0.t
# CHECK-INST: vlse16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 08b55407 <unknown>

vlse16.v v8, (a0), a1
# CHECK-INST: vlse16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0ab55407 <unknown>

vlse32.v v8, (a0), a1, v0.t
# CHECK-INST: vlse32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 08b56407 <unknown>

vlse32.v v8, (a0), a1
# CHECK-INST: vlse32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0ab56407 <unknown>

vlse64.v v8, (a0), a1, v0.t
# CHECK-INST: vlse64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 08b57407 <unknown>

vlse64.v v8, (a0), a1
# CHECK-INST: vlse64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x0a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0ab57407 <unknown>

vluxei8.v v8, (a0), v4, v0.t
# CHECK-INST: vluxei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x04]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 04450407 <unknown>

vluxei8.v v8, (a0), v4
# CHECK-INST: vluxei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x06]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 06450407 <unknown>

vluxei16.v v8, (a0), v4, v0.t
# CHECK-INST: vluxei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x04]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 04455407 <unknown>

vluxei16.v v8, (a0), v4
# CHECK-INST: vluxei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x06]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 06455407 <unknown>

vluxei32.v v8, (a0), v4, v0.t
# CHECK-INST: vluxei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x04]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 04456407 <unknown>

vluxei32.v v8, (a0), v4
# CHECK-INST: vluxei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x06]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 06456407 <unknown>

vluxei64.v v8, (a0), v4, v0.t
# CHECK-INST: vluxei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x04]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 04457407 <unknown>

vluxei64.v v8, (a0), v4
# CHECK-INST: vluxei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x06]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 06457407 <unknown>

vloxei8.v v8, (a0), v4, v0.t
# CHECK-INST: vloxei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0c450407 <unknown>

vloxei8.v v8, (a0), v4
# CHECK-INST: vloxei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0e450407 <unknown>

vloxei16.v v8, (a0), v4, v0.t
# CHECK-INST: vloxei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0c455407 <unknown>

vloxei16.v v8, (a0), v4
# CHECK-INST: vloxei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0e455407 <unknown>

vloxei32.v v8, (a0), v4, v0.t
# CHECK-INST: vloxei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0c456407 <unknown>

vloxei32.v v8, (a0), v4
# CHECK-INST: vloxei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0e456407 <unknown>

vloxei64.v v8, (a0), v4, v0.t
# CHECK-INST: vloxei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x0c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0c457407 <unknown>

vloxei64.v v8, (a0), v4
# CHECK-INST: vloxei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0e457407 <unknown>

vl1re8.v v8, (a0)
# CHECK-INST: vl1re8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x85,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 02850407 <unknown>

vl1re16.v v8, (a0)
# CHECK-INST: vl1re16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x85,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 02855407 <unknown>

vl1re32.v v8, (a0)
# CHECK-INST: vl1re32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x85,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 02856407 <unknown>

vl1re64.v v8, (a0)
# CHECK-INST: vl1re64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x85,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 02857407 <unknown>

vl2re8.v v8, (a0)
# CHECK-INST: vl2re8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x85,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 22850407 <unknown>

vl2re16.v v8, (a0)
# CHECK-INST: vl2re16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x85,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 22855407 <unknown>

vl2re32.v v8, (a0)
# CHECK-INST: vl2re32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x85,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 22856407 <unknown>

vl2re64.v v8, (a0)
# CHECK-INST: vl2re64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x85,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 22857407 <unknown>

vl4re8.v v8, (a0)
# CHECK-INST: vl4re8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x85,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 62850407 <unknown>

vl4re16.v v8, (a0)
# CHECK-INST: vl4re16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x85,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 62855407 <unknown>

vl4re32.v v8, (a0)
# CHECK-INST: vl4re32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x85,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 62856407 <unknown>

vl4re64.v v8, (a0)
# CHECK-INST: vl4re64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x85,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 62857407 <unknown>

vl8re8.v v8, (a0)
# CHECK-INST: vl8re8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x85,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e2850407 <unknown>

vl8re16.v v8, (a0)
# CHECK-INST: vl8re16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x85,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e2855407 <unknown>

vl8re32.v v8, (a0)
# CHECK-INST: vl8re32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x85,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e2856407 <unknown>

vl8re64.v v8, (a0)
# CHECK-INST: vl8re64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x85,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e2857407 <unknown>

vlm.v v0, 0(a0)
# CHECK-INST: vlm.v v0, (a0)
# CHECK-ENCODING: [0x07,0x00,0xb5,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 02b50007 <unknown>

vle8.v v8, 0(a0)
# CHECK-INST: vle8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x02]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 02050407 <unknown>

vle8ff.v v8, 0(a0), v0.t
# CHECK-INST: vle8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x01]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 01050407 <unknown>

vlse16.v v8, 0(a0), a1, v0.t
# CHECK-INST: vlse16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x08]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 08b55407 <unknown>

vluxei32.v v8, 0(a0), v4
# CHECK-INST: vluxei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x06]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 06456407 <unknown>

vloxei64.v v8, 0(a0), v4
# CHECK-INST: vloxei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x0e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 0e457407 <unknown>
