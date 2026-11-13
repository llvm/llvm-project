# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:   --M no-aliases | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:   | llvm-objdump -d --mattr=+v -M no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST

vsm.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsm.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x02]

vse8.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vse8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x00]

vse8.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vse8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x02]

vse16.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vse16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x00]

vse16.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vse16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x02]

vse32.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vse32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x00]

vse32.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vse32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x02]

vse64.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vse64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x00]

vse64.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vse64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x02]

vsse8.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsse8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x08]

vsse8.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsse8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x0a]

vsse16.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsse16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x08]

vsse16.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsse16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x0a]

vsse32.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsse32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x08]

vsse32.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsse32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x0a]

vsse64.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsse64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x08]

vsse64.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsse64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x0a]

vsuxei8.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x04]

vsuxei8.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x06]

vsuxei16.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x04]

vsuxei16.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x06]

vsuxei32.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x04]

vsuxei32.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x06]

vsuxei64.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x04]

vsuxei64.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x06]

vsoxei8.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x0c]

vsoxei8.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x0e]

vsoxei16.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x0c]

vsoxei16.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x0e]

vsoxei32.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x0c]

vsoxei32.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x0e]

vsoxei64.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x0c]

vsoxei64.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x0e]

vs1r.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vs1r.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x85,0x02]

vs2r.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vs2r.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x85,0x22]

vs4r.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vs4r.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x85,0x62]

vs8r.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vs8r.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x85,0xe2]

vsm.v v24, 0(a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsm.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x02]

vse8.v v24, 0(a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vse8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x00]

vsse16.v v24, 0(a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsse16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x08]

vsuxei8.v v24, 0(a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x04]

vsoxei32.v v24, 0(a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x0c]
