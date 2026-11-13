# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:   --M no-aliases | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:   | llvm-objdump -d --mattr=+v -M no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST

vlm.v v0, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlm.v v0, (a0)
# CHECK-ENCODING: [0x07,0x00,0xb5,0x02]

vlm.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlm.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0xb5,0x02]

vle8.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x00]

vle8.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x02]

vle16.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x00]

vle16.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x02]

vle32.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x00]

vle32.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x02]

vle64.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x00]

vle64.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x02]

vle8ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x01]

vle8ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x03]

vle16ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x01]

vle16ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x03]

vle32ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x01]

vle32ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x03]

vle64ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x01]

vle64ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x03]

vlse8.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlse8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x08]

vlse8.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlse8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x0a]

vlse16.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlse16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x08]

vlse16.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlse16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x0a]

vlse32.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlse32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x08]

vlse32.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlse32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x0a]

vlse64.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlse64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x08]

vlse64.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlse64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x0a]

vluxei8.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x04]

vluxei8.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x06]

vluxei16.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x04]

vluxei16.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x06]

vluxei32.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x04]

vluxei32.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x06]

vluxei64.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x04]

vluxei64.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x06]

vloxei8.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x0c]

vloxei8.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x0e]

vloxei16.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x0c]

vloxei16.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x0e]

vloxei32.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x0c]

vloxei32.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x0e]

vloxei64.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x0c]

vloxei64.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x0e]

vl1re8.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vl1re8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x85,0x02]

vl1re16.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vl1re16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x85,0x02]

vl1re32.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vl1re32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x85,0x02]

vl1re64.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vl1re64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x85,0x02]

vl2re8.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vl2re8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x85,0x22]

vl2re16.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vl2re16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x85,0x22]

vl2re32.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vl2re32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x85,0x22]

vl2re64.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vl2re64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x85,0x22]

vl4re8.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vl4re8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x85,0x62]

vl4re16.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vl4re16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x85,0x62]

vl4re32.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vl4re32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x85,0x62]

vl4re64.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vl4re64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x85,0x62]

vl8re8.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vl8re8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x85,0xe2]

vl8re16.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vl8re16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x85,0xe2]

vl8re32.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vl8re32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x85,0xe2]

vl8re64.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vl8re64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x85,0xe2]

vlm.v v0, 0(a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlm.v v0, (a0)
# CHECK-ENCODING: [0x07,0x00,0xb5,0x02]

vle8.v v8, 0(a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x02]

vle8ff.v v8, 0(a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vle8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x01]

vlse16.v v8, 0(a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlse16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x08]

vluxei32.v v8, 0(a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x06]

vloxei64.v v8, 0(a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x0e]
