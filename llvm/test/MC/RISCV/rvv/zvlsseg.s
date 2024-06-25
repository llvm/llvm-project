# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:   --riscv-no-aliases \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:   | llvm-objdump -d --mattr=+v -M no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:   | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vlseg2e8.v v8, (a0), v0.t
# CHECK-INST: vlseg2e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 20050407 <unknown>

vlseg2e8.v v8, (a0)
# CHECK-INST: vlseg2e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 22050407 <unknown>

vlseg2e16.v v8, (a0), v0.t
# CHECK-INST: vlseg2e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 20055407 <unknown>

vlseg2e16.v v8, (a0)
# CHECK-INST: vlseg2e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 22055407 <unknown>

vlseg2e32.v v8, (a0), v0.t
# CHECK-INST: vlseg2e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 20056407 <unknown>

vlseg2e32.v v8, (a0)
# CHECK-INST: vlseg2e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 22056407 <unknown>

vlseg2e64.v v8, (a0), v0.t
# CHECK-INST: vlseg2e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 20057407 <unknown>

vlseg2e64.v v8, (a0)
# CHECK-INST: vlseg2e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 22057407 <unknown>

vlseg2e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg2e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x21]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 21050407 <unknown>

vlseg2e8ff.v v8, (a0)
# CHECK-INST: vlseg2e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 23050407 <unknown>

vlseg2e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg2e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x21]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 21055407 <unknown>

vlseg2e16ff.v v8, (a0)
# CHECK-INST: vlseg2e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 23055407 <unknown>

vlseg2e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg2e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x21]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 21056407 <unknown>

vlseg2e32ff.v v8, (a0)
# CHECK-INST: vlseg2e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 23056407 <unknown>

vlseg2e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg2e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x21]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 21057407 <unknown>

vlseg2e64ff.v v8, (a0)
# CHECK-INST: vlseg2e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 23057407 <unknown>

vlsseg2e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg2e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 28b50407 <unknown>

vlsseg2e8.v v8, (a0), a1
# CHECK-INST: vlsseg2e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2ab50407 <unknown>

vlsseg2e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg2e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 28b55407 <unknown>

vlsseg2e16.v v8, (a0), a1
# CHECK-INST: vlsseg2e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2ab55407 <unknown>

vlsseg2e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg2e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 28b56407 <unknown>

vlsseg2e32.v v8, (a0), a1
# CHECK-INST: vlsseg2e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2ab56407 <unknown>

vlsseg2e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg2e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 28b57407 <unknown>

vlsseg2e64.v v8, (a0), a1
# CHECK-INST: vlsseg2e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2ab57407 <unknown>

vluxseg2ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg2ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 24450407 <unknown>

vluxseg2ei8.v v8, (a0), v4
# CHECK-INST: vluxseg2ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 26450407 <unknown>

vluxseg2ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg2ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 24455407 <unknown>

vluxseg2ei16.v v8, (a0), v4
# CHECK-INST: vluxseg2ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 26455407 <unknown>

vluxseg2ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg2ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 24456407 <unknown>

vluxseg2ei32.v v8, (a0), v4
# CHECK-INST: vluxseg2ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 26456407 <unknown>

vluxseg2ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg2ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 24457407 <unknown>

vluxseg2ei64.v v8, (a0), v4
# CHECK-INST: vluxseg2ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 26457407 <unknown>

vloxseg2ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg2ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2c450407 <unknown>

vloxseg2ei8.v v8, (a0), v4
# CHECK-INST: vloxseg2ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2e450407 <unknown>

vloxseg2ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg2ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2c455407 <unknown>

vloxseg2ei16.v v8, (a0), v4
# CHECK-INST: vloxseg2ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2e455407 <unknown>

vloxseg2ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg2ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2c456407 <unknown>

vloxseg2ei32.v v8, (a0), v4
# CHECK-INST: vloxseg2ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2e456407 <unknown>

vloxseg2ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg2ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2c457407 <unknown>

vloxseg2ei64.v v8, (a0), v4
# CHECK-INST: vloxseg2ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2e457407 <unknown>

vlseg3e8.v v8, (a0), v0.t
# CHECK-INST: vlseg3e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 40050407 <unknown>

vlseg3e8.v v8, (a0)
# CHECK-INST: vlseg3e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 42050407 <unknown>

vlseg3e16.v v8, (a0), v0.t
# CHECK-INST: vlseg3e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 40055407 <unknown>

vlseg3e16.v v8, (a0)
# CHECK-INST: vlseg3e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 42055407 <unknown>

vlseg3e32.v v8, (a0), v0.t
# CHECK-INST: vlseg3e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 40056407 <unknown>

vlseg3e32.v v8, (a0)
# CHECK-INST: vlseg3e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 42056407 <unknown>

vlseg3e64.v v8, (a0), v0.t
# CHECK-INST: vlseg3e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 40057407 <unknown>

vlseg3e64.v v8, (a0)
# CHECK-INST: vlseg3e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 42057407 <unknown>

vlseg3e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg3e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x41]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 41050407 <unknown>

vlseg3e8ff.v v8, (a0)
# CHECK-INST: vlseg3e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x43]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 43050407 <unknown>

vlseg3e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg3e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x41]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 41055407 <unknown>

vlseg3e16ff.v v8, (a0)
# CHECK-INST: vlseg3e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x43]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 43055407 <unknown>

vlseg3e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg3e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x41]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 41056407 <unknown>

vlseg3e32ff.v v8, (a0)
# CHECK-INST: vlseg3e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x43]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 43056407 <unknown>

vlseg3e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg3e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x41]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 41057407 <unknown>

vlseg3e64ff.v v8, (a0)
# CHECK-INST: vlseg3e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x43]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 43057407 <unknown>

vlsseg3e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg3e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48b50407 <unknown>

vlsseg3e8.v v8, (a0), a1
# CHECK-INST: vlsseg3e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4ab50407 <unknown>

vlsseg3e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg3e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48b55407 <unknown>

vlsseg3e16.v v8, (a0), a1
# CHECK-INST: vlsseg3e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4ab55407 <unknown>

vlsseg3e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg3e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48b56407 <unknown>

vlsseg3e32.v v8, (a0), a1
# CHECK-INST: vlsseg3e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4ab56407 <unknown>

vlsseg3e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg3e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48b57407 <unknown>

vlsseg3e64.v v8, (a0), a1
# CHECK-INST: vlsseg3e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4ab57407 <unknown>

vluxseg3ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg3ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 44450407 <unknown>

vluxseg3ei8.v v8, (a0), v4
# CHECK-INST: vluxseg3ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 46450407 <unknown>

vluxseg3ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg3ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 44455407 <unknown>

vluxseg3ei16.v v8, (a0), v4
# CHECK-INST: vluxseg3ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 46455407 <unknown>

vluxseg3ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg3ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 44456407 <unknown>

vluxseg3ei32.v v8, (a0), v4
# CHECK-INST: vluxseg3ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 46456407 <unknown>

vluxseg3ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg3ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 44457407 <unknown>

vluxseg3ei64.v v8, (a0), v4
# CHECK-INST: vluxseg3ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 46457407 <unknown>

vloxseg3ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg3ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4c450407 <unknown>

vloxseg3ei8.v v8, (a0), v4
# CHECK-INST: vloxseg3ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4e450407 <unknown>

vloxseg3ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg3ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4c455407 <unknown>

vloxseg3ei16.v v8, (a0), v4
# CHECK-INST: vloxseg3ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4e455407 <unknown>

vloxseg3ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg3ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4c456407 <unknown>

vloxseg3ei32.v v8, (a0), v4
# CHECK-INST: vloxseg3ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4e456407 <unknown>

vloxseg3ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg3ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4c457407 <unknown>

vloxseg3ei64.v v8, (a0), v4
# CHECK-INST: vloxseg3ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4e457407 <unknown>

vlseg4e8.v v8, (a0), v0.t
# CHECK-INST: vlseg4e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 60050407 <unknown>

vlseg4e8.v v8, (a0)
# CHECK-INST: vlseg4e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 62050407 <unknown>

vlseg4e16.v v8, (a0), v0.t
# CHECK-INST: vlseg4e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 60055407 <unknown>

vlseg4e16.v v8, (a0)
# CHECK-INST: vlseg4e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 62055407 <unknown>

vlseg4e32.v v8, (a0), v0.t
# CHECK-INST: vlseg4e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 60056407 <unknown>

vlseg4e32.v v8, (a0)
# CHECK-INST: vlseg4e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 62056407 <unknown>

vlseg4e64.v v8, (a0), v0.t
# CHECK-INST: vlseg4e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 60057407 <unknown>

vlseg4e64.v v8, (a0)
# CHECK-INST: vlseg4e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 62057407 <unknown>

vlseg4e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg4e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x61]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 61050407 <unknown>

vlseg4e8ff.v v8, (a0)
# CHECK-INST: vlseg4e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x63]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 63050407 <unknown>

vlseg4e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg4e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x61]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 61055407 <unknown>

vlseg4e16ff.v v8, (a0)
# CHECK-INST: vlseg4e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x63]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 63055407 <unknown>

vlseg4e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg4e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x61]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 61056407 <unknown>

vlseg4e32ff.v v8, (a0)
# CHECK-INST: vlseg4e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x63]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 63056407 <unknown>

vlseg4e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg4e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x61]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 61057407 <unknown>

vlseg4e64ff.v v8, (a0)
# CHECK-INST: vlseg4e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x63]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 63057407 <unknown>

vlsseg4e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg4e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 68b50407 <unknown>

vlsseg4e8.v v8, (a0), a1
# CHECK-INST: vlsseg4e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6ab50407 <unknown>

vlsseg4e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg4e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 68b55407 <unknown>

vlsseg4e16.v v8, (a0), a1
# CHECK-INST: vlsseg4e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6ab55407 <unknown>

vlsseg4e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg4e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 68b56407 <unknown>

vlsseg4e32.v v8, (a0), a1
# CHECK-INST: vlsseg4e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6ab56407 <unknown>

vlsseg4e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg4e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 68b57407 <unknown>

vlsseg4e64.v v8, (a0), a1
# CHECK-INST: vlsseg4e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6ab57407 <unknown>

vluxseg4ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg4ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 64450407 <unknown>

vluxseg4ei8.v v8, (a0), v4
# CHECK-INST: vluxseg4ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 66450407 <unknown>

vluxseg4ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg4ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 64455407 <unknown>

vluxseg4ei16.v v8, (a0), v4
# CHECK-INST: vluxseg4ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 66455407 <unknown>

vluxseg4ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg4ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 64456407 <unknown>

vluxseg4ei32.v v8, (a0), v4
# CHECK-INST: vluxseg4ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 66456407 <unknown>

vluxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 64457407 <unknown>

vluxseg4ei64.v v8, (a0), v4
# CHECK-INST: vluxseg4ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 66457407 <unknown>

vloxseg4ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg4ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c450407 <unknown>

vloxseg4ei8.v v8, (a0), v4
# CHECK-INST: vloxseg4ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e450407 <unknown>

vloxseg4ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg4ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c455407 <unknown>

vloxseg4ei16.v v8, (a0), v4
# CHECK-INST: vloxseg4ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e455407 <unknown>

vloxseg4ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg4ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c456407 <unknown>

vloxseg4ei32.v v8, (a0), v4
# CHECK-INST: vloxseg4ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e456407 <unknown>

vloxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c457407 <unknown>

vloxseg4ei64.v v8, (a0), v4
# CHECK-INST: vloxseg4ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e457407 <unknown>

vlseg5e8.v v8, (a0), v0.t
# CHECK-INST: vlseg5e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 80050407 <unknown>

vlseg5e8.v v8, (a0)
# CHECK-INST: vlseg5e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 82050407 <unknown>

vlseg5e16.v v8, (a0), v0.t
# CHECK-INST: vlseg5e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 80055407 <unknown>

vlseg5e16.v v8, (a0)
# CHECK-INST: vlseg5e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 82055407 <unknown>

vlseg5e32.v v8, (a0), v0.t
# CHECK-INST: vlseg5e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 80056407 <unknown>

vlseg5e32.v v8, (a0)
# CHECK-INST: vlseg5e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 82056407 <unknown>

vlseg5e64.v v8, (a0), v0.t
# CHECK-INST: vlseg5e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 80057407 <unknown>

vlseg5e64.v v8, (a0)
# CHECK-INST: vlseg5e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 82057407 <unknown>

vlseg5e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg5e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x81]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 81050407 <unknown>

vlseg5e8ff.v v8, (a0)
# CHECK-INST: vlseg5e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x83]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 83050407 <unknown>

vlseg5e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg5e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x81]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 81055407 <unknown>

vlseg5e16ff.v v8, (a0)
# CHECK-INST: vlseg5e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x83]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 83055407 <unknown>

vlseg5e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg5e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x81]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 81056407 <unknown>

vlseg5e32ff.v v8, (a0)
# CHECK-INST: vlseg5e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x83]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 83056407 <unknown>

vlseg5e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg5e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x81]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 81057407 <unknown>

vlseg5e64ff.v v8, (a0)
# CHECK-INST: vlseg5e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x83]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 83057407 <unknown>

vlsseg5e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg5e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 88b50407 <unknown>

vlsseg5e8.v v8, (a0), a1
# CHECK-INST: vlsseg5e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8ab50407 <unknown>

vlsseg5e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg5e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 88b55407 <unknown>

vlsseg5e16.v v8, (a0), a1
# CHECK-INST: vlsseg5e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8ab55407 <unknown>

vlsseg5e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg5e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 88b56407 <unknown>

vlsseg5e32.v v8, (a0), a1
# CHECK-INST: vlsseg5e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8ab56407 <unknown>

vlsseg5e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg5e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 88b57407 <unknown>

vlsseg5e64.v v8, (a0), a1
# CHECK-INST: vlsseg5e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8ab57407 <unknown>

vluxseg5ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg5ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 84450407 <unknown>

vluxseg5ei8.v v8, (a0), v4
# CHECK-INST: vluxseg5ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 86450407 <unknown>

vluxseg5ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg5ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 84455407 <unknown>

vluxseg5ei16.v v8, (a0), v4
# CHECK-INST: vluxseg5ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 86455407 <unknown>

vluxseg5ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg5ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 84456407 <unknown>

vluxseg5ei32.v v8, (a0), v4
# CHECK-INST: vluxseg5ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 86456407 <unknown>

vluxseg5ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg5ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 84457407 <unknown>

vluxseg5ei64.v v8, (a0), v4
# CHECK-INST: vluxseg5ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 86457407 <unknown>

vloxseg5ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg5ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8c450407 <unknown>

vloxseg5ei8.v v8, (a0), v4
# CHECK-INST: vloxseg5ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8e450407 <unknown>

vloxseg5ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg5ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8c455407 <unknown>

vloxseg5ei16.v v8, (a0), v4
# CHECK-INST: vloxseg5ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8e455407 <unknown>

vloxseg5ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg5ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8c456407 <unknown>

vloxseg5ei32.v v8, (a0), v4
# CHECK-INST: vloxseg5ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8e456407 <unknown>

vloxseg5ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg5ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8c457407 <unknown>

vloxseg5ei64.v v8, (a0), v4
# CHECK-INST: vloxseg5ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8e457407 <unknown>

vlseg6e8.v v8, (a0), v0.t
# CHECK-INST: vlseg6e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a0050407 <unknown>

vlseg6e8.v v8, (a0)
# CHECK-INST: vlseg6e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a2050407 <unknown>

vlseg6e16.v v8, (a0), v0.t
# CHECK-INST: vlseg6e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a0055407 <unknown>

vlseg6e16.v v8, (a0)
# CHECK-INST: vlseg6e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a2055407 <unknown>

vlseg6e32.v v8, (a0), v0.t
# CHECK-INST: vlseg6e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a0056407 <unknown>

vlseg6e32.v v8, (a0)
# CHECK-INST: vlseg6e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a2056407 <unknown>

vlseg6e64.v v8, (a0), v0.t
# CHECK-INST: vlseg6e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a0057407 <unknown>

vlseg6e64.v v8, (a0)
# CHECK-INST: vlseg6e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a2057407 <unknown>

vlseg6e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg6e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xa1]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a1050407 <unknown>

vlseg6e8ff.v v8, (a0)
# CHECK-INST: vlseg6e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xa3]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a3050407 <unknown>

vlseg6e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg6e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xa1]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a1055407 <unknown>

vlseg6e16ff.v v8, (a0)
# CHECK-INST: vlseg6e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xa3]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a3055407 <unknown>

vlseg6e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg6e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xa1]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a1056407 <unknown>

vlseg6e32ff.v v8, (a0)
# CHECK-INST: vlseg6e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xa3]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a3056407 <unknown>

vlseg6e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg6e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xa1]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a1057407 <unknown>

vlseg6e64ff.v v8, (a0)
# CHECK-INST: vlseg6e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xa3]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a3057407 <unknown>

vlsseg6e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg6e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a8b50407 <unknown>

vlsseg6e8.v v8, (a0), a1
# CHECK-INST: vlsseg6e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: aab50407 <unknown>

vlsseg6e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg6e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a8b55407 <unknown>

vlsseg6e16.v v8, (a0), a1
# CHECK-INST: vlsseg6e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: aab55407 <unknown>

vlsseg6e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg6e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a8b56407 <unknown>

vlsseg6e32.v v8, (a0), a1
# CHECK-INST: vlsseg6e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: aab56407 <unknown>

vlsseg6e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg6e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a8b57407 <unknown>

vlsseg6e64.v v8, (a0), a1
# CHECK-INST: vlsseg6e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: aab57407 <unknown>

vluxseg6ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg6ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a4450407 <unknown>

vluxseg6ei8.v v8, (a0), v4
# CHECK-INST: vluxseg6ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a6450407 <unknown>

vluxseg6ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg6ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a4455407 <unknown>

vluxseg6ei16.v v8, (a0), v4
# CHECK-INST: vluxseg6ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a6455407 <unknown>

vluxseg6ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg6ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a4456407 <unknown>

vluxseg6ei32.v v8, (a0), v4
# CHECK-INST: vluxseg6ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a6456407 <unknown>

vluxseg6ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg6ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a4457407 <unknown>

vluxseg6ei64.v v8, (a0), v4
# CHECK-INST: vluxseg6ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a6457407 <unknown>

vloxseg6ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg6ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ac450407 <unknown>

vloxseg6ei8.v v8, (a0), v4
# CHECK-INST: vloxseg6ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ae450407 <unknown>

vloxseg6ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg6ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ac455407 <unknown>

vloxseg6ei16.v v8, (a0), v4
# CHECK-INST: vloxseg6ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ae455407 <unknown>

vloxseg6ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg6ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ac456407 <unknown>

vloxseg6ei32.v v8, (a0), v4
# CHECK-INST: vloxseg6ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ae456407 <unknown>

vloxseg6ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg6ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ac457407 <unknown>

vloxseg6ei64.v v8, (a0), v4
# CHECK-INST: vloxseg6ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ae457407 <unknown>

vlseg7e8.v v8, (a0), v0.t
# CHECK-INST: vlseg7e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c0050407 <unknown>

vlseg7e8.v v8, (a0)
# CHECK-INST: vlseg7e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c2050407 <unknown>

vlseg7e16.v v8, (a0), v0.t
# CHECK-INST: vlseg7e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c0055407 <unknown>

vlseg7e16.v v8, (a0)
# CHECK-INST: vlseg7e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c2055407 <unknown>

vlseg7e32.v v8, (a0), v0.t
# CHECK-INST: vlseg7e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c0056407 <unknown>

vlseg7e32.v v8, (a0)
# CHECK-INST: vlseg7e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c2056407 <unknown>

vlseg7e64.v v8, (a0), v0.t
# CHECK-INST: vlseg7e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c0057407 <unknown>

vlseg7e64.v v8, (a0)
# CHECK-INST: vlseg7e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c2057407 <unknown>

vlseg7e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg7e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xc1]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c1050407 <unknown>

vlseg7e8ff.v v8, (a0)
# CHECK-INST: vlseg7e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xc3]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c3050407 <unknown>

vlseg7e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg7e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xc1]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c1055407 <unknown>

vlseg7e16ff.v v8, (a0)
# CHECK-INST: vlseg7e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xc3]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c3055407 <unknown>

vlseg7e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg7e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xc1]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c1056407 <unknown>

vlseg7e32ff.v v8, (a0)
# CHECK-INST: vlseg7e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xc3]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c3056407 <unknown>

vlseg7e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg7e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xc1]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c1057407 <unknown>

vlseg7e64ff.v v8, (a0)
# CHECK-INST: vlseg7e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xc3]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c3057407 <unknown>

vlsseg7e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg7e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c8b50407 <unknown>

vlsseg7e8.v v8, (a0), a1
# CHECK-INST: vlsseg7e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cab50407 <unknown>

vlsseg7e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg7e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c8b55407 <unknown>

vlsseg7e16.v v8, (a0), a1
# CHECK-INST: vlsseg7e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cab55407 <unknown>

vlsseg7e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg7e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c8b56407 <unknown>

vlsseg7e32.v v8, (a0), a1
# CHECK-INST: vlsseg7e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cab56407 <unknown>

vlsseg7e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg7e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c8b57407 <unknown>

vlsseg7e64.v v8, (a0), a1
# CHECK-INST: vlsseg7e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cab57407 <unknown>

vluxseg7ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg7ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c4450407 <unknown>

vluxseg7ei8.v v8, (a0), v4
# CHECK-INST: vluxseg7ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c6450407 <unknown>

vluxseg7ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg7ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c4455407 <unknown>

vluxseg7ei16.v v8, (a0), v4
# CHECK-INST: vluxseg7ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c6455407 <unknown>

vluxseg7ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg7ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c4456407 <unknown>

vluxseg7ei32.v v8, (a0), v4
# CHECK-INST: vluxseg7ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c6456407 <unknown>

vluxseg7ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg7ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c4457407 <unknown>

vluxseg7ei64.v v8, (a0), v4
# CHECK-INST: vluxseg7ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c6457407 <unknown>

vloxseg7ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg7ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cc450407 <unknown>

vloxseg7ei8.v v8, (a0), v4
# CHECK-INST: vloxseg7ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ce450407 <unknown>

vloxseg7ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg7ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cc455407 <unknown>

vloxseg7ei16.v v8, (a0), v4
# CHECK-INST: vloxseg7ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ce455407 <unknown>

vloxseg7ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg7ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cc456407 <unknown>

vloxseg7ei32.v v8, (a0), v4
# CHECK-INST: vloxseg7ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ce456407 <unknown>

vloxseg7ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg7ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cc457407 <unknown>

vloxseg7ei64.v v8, (a0), v4
# CHECK-INST: vloxseg7ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ce457407 <unknown>

vlseg8e8.v v8, (a0), v0.t
# CHECK-INST: vlseg8e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e0050407 <unknown>

vlseg8e8.v v8, (a0)
# CHECK-INST: vlseg8e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e2050407 <unknown>

vlseg8e16.v v8, (a0), v0.t
# CHECK-INST: vlseg8e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e0055407 <unknown>

vlseg8e16.v v8, (a0)
# CHECK-INST: vlseg8e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e2055407 <unknown>

vlseg8e32.v v8, (a0), v0.t
# CHECK-INST: vlseg8e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e0056407 <unknown>

vlseg8e32.v v8, (a0)
# CHECK-INST: vlseg8e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e2056407 <unknown>

vlseg8e64.v v8, (a0), v0.t
# CHECK-INST: vlseg8e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e0057407 <unknown>

vlseg8e64.v v8, (a0)
# CHECK-INST: vlseg8e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e2057407 <unknown>

vlseg8e8ff.v v8, (a0), v0.t
# CHECK-INST: vlseg8e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xe1]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e1050407 <unknown>

vlseg8e8ff.v v8, (a0)
# CHECK-INST: vlseg8e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xe3]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e3050407 <unknown>

vlseg8e16ff.v v8, (a0), v0.t
# CHECK-INST: vlseg8e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xe1]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e1055407 <unknown>

vlseg8e16ff.v v8, (a0)
# CHECK-INST: vlseg8e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xe3]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e3055407 <unknown>

vlseg8e32ff.v v8, (a0), v0.t
# CHECK-INST: vlseg8e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xe1]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e1056407 <unknown>

vlseg8e32ff.v v8, (a0)
# CHECK-INST: vlseg8e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xe3]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e3056407 <unknown>

vlseg8e64ff.v v8, (a0), v0.t
# CHECK-INST: vlseg8e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xe1]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e1057407 <unknown>

vlseg8e64ff.v v8, (a0)
# CHECK-INST: vlseg8e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xe3]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e3057407 <unknown>

vlsseg8e8.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg8e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e8b50407 <unknown>

vlsseg8e8.v v8, (a0), a1
# CHECK-INST: vlsseg8e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: eab50407 <unknown>

vlsseg8e16.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg8e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e8b55407 <unknown>

vlsseg8e16.v v8, (a0), a1
# CHECK-INST: vlsseg8e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: eab55407 <unknown>

vlsseg8e32.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg8e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e8b56407 <unknown>

vlsseg8e32.v v8, (a0), a1
# CHECK-INST: vlsseg8e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: eab56407 <unknown>

vlsseg8e64.v v8, (a0), a1, v0.t
# CHECK-INST: vlsseg8e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e8b57407 <unknown>

vlsseg8e64.v v8, (a0), a1
# CHECK-INST: vlsseg8e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: eab57407 <unknown>

vluxseg8ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg8ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e4450407 <unknown>

vluxseg8ei8.v v8, (a0), v4
# CHECK-INST: vluxseg8ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e6450407 <unknown>

vluxseg8ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg8ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e4455407 <unknown>

vluxseg8ei16.v v8, (a0), v4
# CHECK-INST: vluxseg8ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e6455407 <unknown>

vluxseg8ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg8ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e4456407 <unknown>

vluxseg8ei32.v v8, (a0), v4
# CHECK-INST: vluxseg8ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e6456407 <unknown>

vluxseg8ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vluxseg8ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e4457407 <unknown>

vluxseg8ei64.v v8, (a0), v4
# CHECK-INST: vluxseg8ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e6457407 <unknown>

vloxseg8ei8.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg8ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ec450407 <unknown>

vloxseg8ei8.v v8, (a0), v4
# CHECK-INST: vloxseg8ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ee450407 <unknown>

vloxseg8ei16.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg8ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ec455407 <unknown>

vloxseg8ei16.v v8, (a0), v4
# CHECK-INST: vloxseg8ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ee455407 <unknown>

vloxseg8ei32.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg8ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ec456407 <unknown>

vloxseg8ei32.v v8, (a0), v4
# CHECK-INST: vloxseg8ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ee456407 <unknown>

vloxseg8ei64.v v8, (a0), v4, v0.t
# CHECK-INST: vloxseg8ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ec457407 <unknown>

vloxseg8ei64.v v8, (a0), v4
# CHECK-INST: vloxseg8ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ee457407 <unknown>

vsseg2e8.v v24, (a0), v0.t
# CHECK-INST: vsseg2e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 20050c27 <unknown>

vsseg2e8.v v24, (a0)
# CHECK-INST: vsseg2e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 22050c27 <unknown>

vsseg2e16.v v24, (a0), v0.t
# CHECK-INST: vsseg2e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 20055c27 <unknown>

vsseg2e16.v v24, (a0)
# CHECK-INST: vsseg2e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 22055c27 <unknown>

vsseg2e32.v v24, (a0), v0.t
# CHECK-INST: vsseg2e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 20056c27 <unknown>

vsseg2e32.v v24, (a0)
# CHECK-INST: vsseg2e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 22056c27 <unknown>

vsseg2e64.v v24, (a0), v0.t
# CHECK-INST: vsseg2e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 20057c27 <unknown>

vsseg2e64.v v24, (a0)
# CHECK-INST: vsseg2e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x22]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 22057c27 <unknown>

vssseg2e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg2e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 28b50c27 <unknown>

vssseg2e8.v v24, (a0), a1
# CHECK-INST: vssseg2e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2ab50c27 <unknown>

vssseg2e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg2e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 28b55c27 <unknown>

vssseg2e16.v v24, (a0), a1
# CHECK-INST: vssseg2e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2ab55c27 <unknown>

vssseg2e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg2e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 28b56c27 <unknown>

vssseg2e32.v v24, (a0), a1
# CHECK-INST: vssseg2e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2ab56c27 <unknown>

vssseg2e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg2e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 28b57c27 <unknown>

vssseg2e64.v v24, (a0), a1
# CHECK-INST: vssseg2e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2ab57c27 <unknown>

vsuxseg2ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg2ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 24450c27 <unknown>

vsuxseg2ei8.v v24, (a0), v4
# CHECK-INST: vsuxseg2ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 26450c27 <unknown>

vsuxseg2ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg2ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 24455c27 <unknown>

vsuxseg2ei16.v v24, (a0), v4
# CHECK-INST: vsuxseg2ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 26455c27 <unknown>

vsuxseg2ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg2ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 24456c27 <unknown>

vsuxseg2ei32.v v24, (a0), v4
# CHECK-INST: vsuxseg2ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 26456c27 <unknown>

vsuxseg2ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg2ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x24]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 24457c27 <unknown>

vsuxseg2ei64.v v24, (a0), v4
# CHECK-INST: vsuxseg2ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x26]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 26457c27 <unknown>

vsoxseg2ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg2ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2c450c27 <unknown>

vsoxseg2ei8.v v24, (a0), v4
# CHECK-INST: vsoxseg2ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2e450c27 <unknown>

vsoxseg2ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg2ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2c455c27 <unknown>

vsoxseg2ei16.v v24, (a0), v4
# CHECK-INST: vsoxseg2ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2e455c27 <unknown>

vsoxseg2ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg2ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2c456c27 <unknown>

vsoxseg2ei32.v v24, (a0), v4
# CHECK-INST: vsoxseg2ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2e456c27 <unknown>

vsoxseg2ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg2ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x2c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2c457c27 <unknown>

vsoxseg2ei64.v v24, (a0), v4
# CHECK-INST: vsoxseg2ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x2e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2e457c27 <unknown>

vsseg3e8.v v24, (a0), v0.t
# CHECK-INST: vsseg3e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 40050c27 <unknown>

vsseg3e8.v v24, (a0)
# CHECK-INST: vsseg3e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 42050c27 <unknown>

vsseg3e16.v v24, (a0), v0.t
# CHECK-INST: vsseg3e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 40055c27 <unknown>

vsseg3e16.v v24, (a0)
# CHECK-INST: vsseg3e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 42055c27 <unknown>

vsseg3e32.v v24, (a0), v0.t
# CHECK-INST: vsseg3e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 40056c27 <unknown>

vsseg3e32.v v24, (a0)
# CHECK-INST: vsseg3e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 42056c27 <unknown>

vsseg3e64.v v24, (a0), v0.t
# CHECK-INST: vsseg3e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x40]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 40057c27 <unknown>

vsseg3e64.v v24, (a0)
# CHECK-INST: vsseg3e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x42]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 42057c27 <unknown>

vssseg3e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg3e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48b50c27 <unknown>

vssseg3e8.v v24, (a0), a1
# CHECK-INST: vssseg3e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4ab50c27 <unknown>

vssseg3e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg3e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48b55c27 <unknown>

vssseg3e16.v v24, (a0), a1
# CHECK-INST: vssseg3e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4ab55c27 <unknown>

vssseg3e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg3e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48b56c27 <unknown>

vssseg3e32.v v24, (a0), a1
# CHECK-INST: vssseg3e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4ab56c27 <unknown>

vssseg3e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg3e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48b57c27 <unknown>

vssseg3e64.v v24, (a0), a1
# CHECK-INST: vssseg3e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4ab57c27 <unknown>

vsuxseg3ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg3ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 44450c27 <unknown>

vsuxseg3ei8.v v24, (a0), v4
# CHECK-INST: vsuxseg3ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 46450c27 <unknown>

vsuxseg3ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg3ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 44455c27 <unknown>

vsuxseg3ei16.v v24, (a0), v4
# CHECK-INST: vsuxseg3ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 46455c27 <unknown>

vsuxseg3ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg3ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 44456c27 <unknown>

vsuxseg3ei32.v v24, (a0), v4
# CHECK-INST: vsuxseg3ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 46456c27 <unknown>

vsuxseg3ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg3ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x44]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 44457c27 <unknown>

vsuxseg3ei64.v v24, (a0), v4
# CHECK-INST: vsuxseg3ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 46457c27 <unknown>

vsoxseg3ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg3ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4c450c27 <unknown>

vsoxseg3ei8.v v24, (a0), v4
# CHECK-INST: vsoxseg3ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4e450c27 <unknown>

vsoxseg3ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg3ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4c455c27 <unknown>

vsoxseg3ei16.v v24, (a0), v4
# CHECK-INST: vsoxseg3ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4e455c27 <unknown>

vsoxseg3ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg3ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4c456c27 <unknown>

vsoxseg3ei32.v v24, (a0), v4
# CHECK-INST: vsoxseg3ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4e456c27 <unknown>

vsoxseg3ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg3ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x4c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4c457c27 <unknown>

vsoxseg3ei64.v v24, (a0), v4
# CHECK-INST: vsoxseg3ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x4e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4e457c27 <unknown>

vsseg4e8.v v24, (a0), v0.t
# CHECK-INST: vsseg4e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 60050c27 <unknown>

vsseg4e8.v v24, (a0)
# CHECK-INST: vsseg4e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 62050c27 <unknown>

vsseg4e16.v v24, (a0), v0.t
# CHECK-INST: vsseg4e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 60055c27 <unknown>

vsseg4e16.v v24, (a0)
# CHECK-INST: vsseg4e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 62055c27 <unknown>

vsseg4e32.v v24, (a0), v0.t
# CHECK-INST: vsseg4e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 60056c27 <unknown>

vsseg4e32.v v24, (a0)
# CHECK-INST: vsseg4e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 62056c27 <unknown>

vsseg4e64.v v24, (a0), v0.t
# CHECK-INST: vsseg4e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x60]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 60057c27 <unknown>

vsseg4e64.v v24, (a0)
# CHECK-INST: vsseg4e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x62]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 62057c27 <unknown>

vssseg4e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg4e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 68b50c27 <unknown>

vssseg4e8.v v24, (a0), a1
# CHECK-INST: vssseg4e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6ab50c27 <unknown>

vssseg4e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg4e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 68b55c27 <unknown>

vssseg4e16.v v24, (a0), a1
# CHECK-INST: vssseg4e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6ab55c27 <unknown>

vssseg4e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg4e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 68b56c27 <unknown>

vssseg4e32.v v24, (a0), a1
# CHECK-INST: vssseg4e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6ab56c27 <unknown>

vssseg4e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg4e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x68]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 68b57c27 <unknown>

vssseg4e64.v v24, (a0), a1
# CHECK-INST: vssseg4e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x6a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6ab57c27 <unknown>

vsuxseg4ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg4ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 64450c27 <unknown>

vsuxseg4ei8.v v24, (a0), v4
# CHECK-INST: vsuxseg4ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 66450c27 <unknown>

vsuxseg4ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg4ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 64455c27 <unknown>

vsuxseg4ei16.v v24, (a0), v4
# CHECK-INST: vsuxseg4ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 66455c27 <unknown>

vsuxseg4ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg4ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 64456c27 <unknown>

vsuxseg4ei32.v v24, (a0), v4
# CHECK-INST: vsuxseg4ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 66456c27 <unknown>

vsuxseg4ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg4ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x64]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 64457c27 <unknown>

vsuxseg4ei64.v v24, (a0), v4
# CHECK-INST: vsuxseg4ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x66]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 66457c27 <unknown>

vsoxseg4ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg4ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c450c27 <unknown>

vsoxseg4ei8.v v24, (a0), v4
# CHECK-INST: vsoxseg4ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e450c27 <unknown>

vsoxseg4ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg4ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c455c27 <unknown>

vsoxseg4ei16.v v24, (a0), v4
# CHECK-INST: vsoxseg4ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e455c27 <unknown>

vsoxseg4ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg4ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c456c27 <unknown>

vsoxseg4ei32.v v24, (a0), v4
# CHECK-INST: vsoxseg4ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e456c27 <unknown>

vsoxseg4ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg4ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c457c27 <unknown>

vsoxseg4ei64.v v24, (a0), v4
# CHECK-INST: vsoxseg4ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x6e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6e457c27 <unknown>

vsseg5e8.v v24, (a0), v0.t
# CHECK-INST: vsseg5e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 80050c27 <unknown>

vsseg5e8.v v24, (a0)
# CHECK-INST: vsseg5e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 82050c27 <unknown>

vsseg5e16.v v24, (a0), v0.t
# CHECK-INST: vsseg5e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 80055c27 <unknown>

vsseg5e16.v v24, (a0)
# CHECK-INST: vsseg5e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 82055c27 <unknown>

vsseg5e32.v v24, (a0), v0.t
# CHECK-INST: vsseg5e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 80056c27 <unknown>

vsseg5e32.v v24, (a0)
# CHECK-INST: vsseg5e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 82056c27 <unknown>

vsseg5e64.v v24, (a0), v0.t
# CHECK-INST: vsseg5e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 80057c27 <unknown>

vsseg5e64.v v24, (a0)
# CHECK-INST: vsseg5e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x82]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 82057c27 <unknown>

vssseg5e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg5e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 88b50c27 <unknown>

vssseg5e8.v v24, (a0), a1
# CHECK-INST: vssseg5e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8ab50c27 <unknown>

vssseg5e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg5e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 88b55c27 <unknown>

vssseg5e16.v v24, (a0), a1
# CHECK-INST: vssseg5e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8ab55c27 <unknown>

vssseg5e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg5e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 88b56c27 <unknown>

vssseg5e32.v v24, (a0), a1
# CHECK-INST: vssseg5e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8ab56c27 <unknown>

vssseg5e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg5e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x88]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 88b57c27 <unknown>

vssseg5e64.v v24, (a0), a1
# CHECK-INST: vssseg5e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x8a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8ab57c27 <unknown>

vsuxseg5ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg5ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 84450c27 <unknown>

vsuxseg5ei8.v v24, (a0), v4
# CHECK-INST: vsuxseg5ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 86450c27 <unknown>

vsuxseg5ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg5ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 84455c27 <unknown>

vsuxseg5ei16.v v24, (a0), v4
# CHECK-INST: vsuxseg5ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 86455c27 <unknown>

vsuxseg5ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg5ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 84456c27 <unknown>

vsuxseg5ei32.v v24, (a0), v4
# CHECK-INST: vsuxseg5ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 86456c27 <unknown>

vsuxseg5ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg5ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x84]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 84457c27 <unknown>

vsuxseg5ei64.v v24, (a0), v4
# CHECK-INST: vsuxseg5ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x86]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 86457c27 <unknown>

vsoxseg5ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg5ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8c450c27 <unknown>

vsoxseg5ei8.v v24, (a0), v4
# CHECK-INST: vsoxseg5ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8e450c27 <unknown>

vsoxseg5ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg5ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8c455c27 <unknown>

vsoxseg5ei16.v v24, (a0), v4
# CHECK-INST: vsoxseg5ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8e455c27 <unknown>

vsoxseg5ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg5ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8c456c27 <unknown>

vsoxseg5ei32.v v24, (a0), v4
# CHECK-INST: vsoxseg5ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8e456c27 <unknown>

vsoxseg5ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg5ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x8c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8c457c27 <unknown>

vsoxseg5ei64.v v24, (a0), v4
# CHECK-INST: vsoxseg5ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x8e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 8e457c27 <unknown>

vsseg6e8.v v24, (a0), v0.t
# CHECK-INST: vsseg6e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a0050c27 <unknown>

vsseg6e8.v v24, (a0)
# CHECK-INST: vsseg6e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a2050c27 <unknown>

vsseg6e16.v v24, (a0), v0.t
# CHECK-INST: vsseg6e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a0055c27 <unknown>

vsseg6e16.v v24, (a0)
# CHECK-INST: vsseg6e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a2055c27 <unknown>

vsseg6e32.v v24, (a0), v0.t
# CHECK-INST: vsseg6e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a0056c27 <unknown>

vsseg6e32.v v24, (a0)
# CHECK-INST: vsseg6e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a2056c27 <unknown>

vsseg6e64.v v24, (a0), v0.t
# CHECK-INST: vsseg6e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0xa0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a0057c27 <unknown>

vsseg6e64.v v24, (a0)
# CHECK-INST: vsseg6e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0xa2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a2057c27 <unknown>

vssseg6e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg6e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a8b50c27 <unknown>

vssseg6e8.v v24, (a0), a1
# CHECK-INST: vssseg6e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: aab50c27 <unknown>

vssseg6e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg6e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a8b55c27 <unknown>

vssseg6e16.v v24, (a0), a1
# CHECK-INST: vssseg6e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: aab55c27 <unknown>

vssseg6e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg6e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a8b56c27 <unknown>

vssseg6e32.v v24, (a0), a1
# CHECK-INST: vssseg6e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: aab56c27 <unknown>

vssseg6e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg6e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xa8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a8b57c27 <unknown>

vssseg6e64.v v24, (a0), a1
# CHECK-INST: vssseg6e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xaa]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: aab57c27 <unknown>

vsuxseg6ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg6ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a4450c27 <unknown>

vsuxseg6ei8.v v24, (a0), v4
# CHECK-INST: vsuxseg6ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a6450c27 <unknown>

vsuxseg6ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg6ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a4455c27 <unknown>

vsuxseg6ei16.v v24, (a0), v4
# CHECK-INST: vsuxseg6ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a6455c27 <unknown>

vsuxseg6ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg6ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a4456c27 <unknown>

vsuxseg6ei32.v v24, (a0), v4
# CHECK-INST: vsuxseg6ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a6456c27 <unknown>

vsuxseg6ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg6ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a4457c27 <unknown>

vsuxseg6ei64.v v24, (a0), v4
# CHECK-INST: vsuxseg6ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xa6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a6457c27 <unknown>

vsoxseg6ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg6ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ac450c27 <unknown>

vsoxseg6ei8.v v24, (a0), v4
# CHECK-INST: vsoxseg6ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ae450c27 <unknown>

vsoxseg6ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg6ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ac455c27 <unknown>

vsoxseg6ei16.v v24, (a0), v4
# CHECK-INST: vsoxseg6ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ae455c27 <unknown>

vsoxseg6ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg6ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ac456c27 <unknown>

vsoxseg6ei32.v v24, (a0), v4
# CHECK-INST: vsoxseg6ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ae456c27 <unknown>

vsoxseg6ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg6ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xac]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ac457c27 <unknown>

vsoxseg6ei64.v v24, (a0), v4
# CHECK-INST: vsoxseg6ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xae]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ae457c27 <unknown>

vsseg7e8.v v24, (a0), v0.t
# CHECK-INST: vsseg7e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c0050c27 <unknown>

vsseg7e8.v v24, (a0)
# CHECK-INST: vsseg7e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c2050c27 <unknown>

vsseg7e16.v v24, (a0), v0.t
# CHECK-INST: vsseg7e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c0055c27 <unknown>

vsseg7e16.v v24, (a0)
# CHECK-INST: vsseg7e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c2055c27 <unknown>

vsseg7e32.v v24, (a0), v0.t
# CHECK-INST: vsseg7e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c0056c27 <unknown>

vsseg7e32.v v24, (a0)
# CHECK-INST: vsseg7e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c2056c27 <unknown>

vsseg7e64.v v24, (a0), v0.t
# CHECK-INST: vsseg7e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0xc0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c0057c27 <unknown>

vsseg7e64.v v24, (a0)
# CHECK-INST: vsseg7e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0xc2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c2057c27 <unknown>

vssseg7e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg7e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c8b50c27 <unknown>

vssseg7e8.v v24, (a0), a1
# CHECK-INST: vssseg7e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cab50c27 <unknown>

vssseg7e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg7e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c8b55c27 <unknown>

vssseg7e16.v v24, (a0), a1
# CHECK-INST: vssseg7e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cab55c27 <unknown>

vssseg7e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg7e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c8b56c27 <unknown>

vssseg7e32.v v24, (a0), a1
# CHECK-INST: vssseg7e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cab56c27 <unknown>

vssseg7e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg7e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xc8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c8b57c27 <unknown>

vssseg7e64.v v24, (a0), a1
# CHECK-INST: vssseg7e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xca]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cab57c27 <unknown>

vsuxseg7ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg7ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c4450c27 <unknown>

vsuxseg7ei8.v v24, (a0), v4
# CHECK-INST: vsuxseg7ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c6450c27 <unknown>

vsuxseg7ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg7ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c4455c27 <unknown>

vsuxseg7ei16.v v24, (a0), v4
# CHECK-INST: vsuxseg7ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c6455c27 <unknown>

vsuxseg7ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg7ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c4456c27 <unknown>

vsuxseg7ei32.v v24, (a0), v4
# CHECK-INST: vsuxseg7ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c6456c27 <unknown>

vsuxseg7ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg7ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xc4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c4457c27 <unknown>

vsuxseg7ei64.v v24, (a0), v4
# CHECK-INST: vsuxseg7ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xc6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: c6457c27 <unknown>

vsoxseg7ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg7ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cc450c27 <unknown>

vsoxseg7ei8.v v24, (a0), v4
# CHECK-INST: vsoxseg7ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ce450c27 <unknown>

vsoxseg7ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg7ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cc455c27 <unknown>

vsoxseg7ei16.v v24, (a0), v4
# CHECK-INST: vsoxseg7ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ce455c27 <unknown>

vsoxseg7ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg7ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cc456c27 <unknown>

vsoxseg7ei32.v v24, (a0), v4
# CHECK-INST: vsoxseg7ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ce456c27 <unknown>

vsoxseg7ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg7ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xcc]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: cc457c27 <unknown>

vsoxseg7ei64.v v24, (a0), v4
# CHECK-INST: vsoxseg7ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ce457c27 <unknown>

vsseg8e8.v v24, (a0), v0.t
# CHECK-INST: vsseg8e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e0050c27 <unknown>

vsseg8e8.v v24, (a0)
# CHECK-INST: vsseg8e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e2050c27 <unknown>

vsseg8e16.v v24, (a0), v0.t
# CHECK-INST: vsseg8e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e0055c27 <unknown>

vsseg8e16.v v24, (a0)
# CHECK-INST: vsseg8e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e2055c27 <unknown>

vsseg8e32.v v24, (a0), v0.t
# CHECK-INST: vsseg8e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e0056c27 <unknown>

vsseg8e32.v v24, (a0)
# CHECK-INST: vsseg8e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e2056c27 <unknown>

vsseg8e64.v v24, (a0), v0.t
# CHECK-INST: vsseg8e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0xe0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e0057c27 <unknown>

vsseg8e64.v v24, (a0)
# CHECK-INST: vsseg8e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e2057c27 <unknown>

vssseg8e8.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg8e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e8b50c27 <unknown>

vssseg8e8.v v24, (a0), a1
# CHECK-INST: vssseg8e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: eab50c27 <unknown>

vssseg8e16.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg8e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e8b55c27 <unknown>

vssseg8e16.v v24, (a0), a1
# CHECK-INST: vssseg8e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: eab55c27 <unknown>

vssseg8e32.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg8e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e8b56c27 <unknown>

vssseg8e32.v v24, (a0), a1
# CHECK-INST: vssseg8e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: eab56c27 <unknown>

vssseg8e64.v v24, (a0), a1, v0.t
# CHECK-INST: vssseg8e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xe8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e8b57c27 <unknown>

vssseg8e64.v v24, (a0), a1
# CHECK-INST: vssseg8e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xea]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: eab57c27 <unknown>

vsuxseg8ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg8ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e4450c27 <unknown>

vsuxseg8ei8.v v24, (a0), v4
# CHECK-INST: vsuxseg8ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e6450c27 <unknown>

vsuxseg8ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg8ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e4455c27 <unknown>

vsuxseg8ei16.v v24, (a0), v4
# CHECK-INST: vsuxseg8ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e6455c27 <unknown>

vsuxseg8ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg8ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e4456c27 <unknown>

vsuxseg8ei32.v v24, (a0), v4
# CHECK-INST: vsuxseg8ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e6456c27 <unknown>

vsuxseg8ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsuxseg8ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xe4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e4457c27 <unknown>

vsuxseg8ei64.v v24, (a0), v4
# CHECK-INST: vsuxseg8ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xe6]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: e6457c27 <unknown>

vsoxseg8ei8.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg8ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ec450c27 <unknown>

vsoxseg8ei8.v v24, (a0), v4
# CHECK-INST: vsoxseg8ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ee450c27 <unknown>

vsoxseg8ei16.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg8ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ec455c27 <unknown>

vsoxseg8ei16.v v24, (a0), v4
# CHECK-INST: vsoxseg8ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ee455c27 <unknown>

vsoxseg8ei32.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg8ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ec456c27 <unknown>

vsoxseg8ei32.v v24, (a0), v4
# CHECK-INST: vsoxseg8ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ee456c27 <unknown>

vsoxseg8ei64.v v24, (a0), v4, v0.t
# CHECK-INST: vsoxseg8ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ec457c27 <unknown>

vsoxseg8ei64.v v24, (a0), v4
# CHECK-INST: vsoxseg8ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ee457c27 <unknown>

vlseg2e8.v v8, 0(a0), v0.t
# CHECK-INST: vlseg2e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x20]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 20050407 <unknown>

vlseg2e16ff.v v8, 0(a0)
# CHECK-INST: vlseg2e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x23]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 23055407 <unknown>

vlsseg2e8.v v8, 0(a0), a1
# CHECK-INST: vlsseg2e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x2a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 2ab50407 <unknown>

vluxseg3ei16.v v8, 0(a0), v4
# CHECK-INST: vluxseg3ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x46]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 46455407 <unknown>

vloxseg4ei64.v v8, 0(a0), v4, v0.t
# CHECK-INST: vloxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x6c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 6c457407 <unknown>

vsseg5e32.v v24, 0(a0), v0.t
# CHECK-INST: vsseg5e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x80]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 80056c27 <unknown>

vssseg2e8.v v24, 0(a0), a1, v0.t
# CHECK-INST: vssseg2e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x28]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 28b50c27 <unknown>

vsoxseg7ei16.v v24, 0(a0), v4
# CHECK-INST: vsoxseg7ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xce]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: ce455c27 <unknown>

vsuxseg6ei32.v v24, 0(a0), v4, v0.t
# CHECK-INST: vsuxseg6ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xa4]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: a4456c27 <unknown>
