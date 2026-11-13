# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:   --M no-aliases \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:   | llvm-objdump -d --mattr=+v -M no-aliases - \
# RUN:   | FileCheck %s --check-prefix=CHECK-INST

vlseg2e8.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x20]

vlseg2e8.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x22]

vlseg2e16.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x20]

vlseg2e16.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x22]

vlseg2e32.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x20]

vlseg2e32.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x22]

vlseg2e64.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x20]

vlseg2e64.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x22]

vlseg2e8ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x21]

vlseg2e8ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x23]

vlseg2e16ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x21]

vlseg2e16ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x23]

vlseg2e32ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x21]

vlseg2e32ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x23]

vlseg2e64ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x21]

vlseg2e64ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x23]

vlsseg2e8.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg2e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x28]

vlsseg2e8.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg2e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x2a]

vlsseg2e16.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg2e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x28]

vlsseg2e16.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg2e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x2a]

vlsseg2e32.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg2e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x28]

vlsseg2e32.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg2e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x2a]

vlsseg2e64.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg2e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x28]

vlsseg2e64.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg2e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x2a]

vluxseg2ei8.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg2ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x24]

vluxseg2ei8.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg2ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x26]

vluxseg2ei16.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg2ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x24]

vluxseg2ei16.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg2ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x26]

vluxseg2ei32.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg2ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x24]

vluxseg2ei32.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg2ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x26]

vluxseg2ei64.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg2ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x24]

vluxseg2ei64.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg2ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x26]

vloxseg2ei8.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg2ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x2c]

vloxseg2ei8.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg2ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x2e]

vloxseg2ei16.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg2ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x2c]

vloxseg2ei16.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg2ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x2e]

vloxseg2ei32.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg2ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x2c]

vloxseg2ei32.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg2ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x2e]

vloxseg2ei64.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg2ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x2c]

vloxseg2ei64.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg2ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x2e]

vlseg3e8.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg3e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x40]

vlseg3e8.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg3e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x42]

vlseg3e16.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg3e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x40]

vlseg3e16.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg3e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x42]

vlseg3e32.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg3e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x40]

vlseg3e32.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg3e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x42]

vlseg3e64.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg3e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x40]

vlseg3e64.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg3e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x42]

vlseg3e8ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg3e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x41]

vlseg3e8ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg3e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x43]

vlseg3e16ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg3e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x41]

vlseg3e16ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg3e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x43]

vlseg3e32ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg3e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x41]

vlseg3e32ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg3e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x43]

vlseg3e64ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg3e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x41]

vlseg3e64ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg3e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x43]

vlsseg3e8.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg3e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x48]

vlsseg3e8.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg3e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x4a]

vlsseg3e16.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg3e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x48]

vlsseg3e16.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg3e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x4a]

vlsseg3e32.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg3e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x48]

vlsseg3e32.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg3e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x4a]

vlsseg3e64.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg3e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x48]

vlsseg3e64.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg3e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x4a]

vluxseg3ei8.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg3ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x44]

vluxseg3ei8.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg3ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x46]

vluxseg3ei16.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg3ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x44]

vluxseg3ei16.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg3ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x46]

vluxseg3ei32.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg3ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x44]

vluxseg3ei32.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg3ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x46]

vluxseg3ei64.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg3ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x44]

vluxseg3ei64.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg3ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x46]

vloxseg3ei8.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg3ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x4c]

vloxseg3ei8.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg3ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x4e]

vloxseg3ei16.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg3ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x4c]

vloxseg3ei16.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg3ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x4e]

vloxseg3ei32.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg3ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x4c]

vloxseg3ei32.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg3ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x4e]

vloxseg3ei64.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg3ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x4c]

vloxseg3ei64.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg3ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x4e]

vlseg4e8.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg4e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x60]

vlseg4e8.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg4e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x62]

vlseg4e16.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg4e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x60]

vlseg4e16.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg4e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x62]

vlseg4e32.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg4e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x60]

vlseg4e32.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg4e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x62]

vlseg4e64.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg4e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x60]

vlseg4e64.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg4e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x62]

vlseg4e8ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg4e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x61]

vlseg4e8ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg4e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x63]

vlseg4e16ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg4e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x61]

vlseg4e16ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg4e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x63]

vlseg4e32ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg4e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x61]

vlseg4e32ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg4e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x63]

vlseg4e64ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg4e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x61]

vlseg4e64ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg4e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x63]

vlsseg4e8.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg4e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x68]

vlsseg4e8.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg4e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x6a]

vlsseg4e16.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg4e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x68]

vlsseg4e16.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg4e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x6a]

vlsseg4e32.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg4e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x68]

vlsseg4e32.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg4e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x6a]

vlsseg4e64.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg4e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x68]

vlsseg4e64.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg4e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x6a]

vluxseg4ei8.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg4ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x64]

vluxseg4ei8.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg4ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x66]

vluxseg4ei16.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg4ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x64]

vluxseg4ei16.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg4ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x66]

vluxseg4ei32.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg4ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x64]

vluxseg4ei32.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg4ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x66]

vluxseg4ei64.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x64]

vluxseg4ei64.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg4ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x66]

vloxseg4ei8.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg4ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x6c]

vloxseg4ei8.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg4ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x6e]

vloxseg4ei16.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg4ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x6c]

vloxseg4ei16.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg4ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x6e]

vloxseg4ei32.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg4ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x6c]

vloxseg4ei32.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg4ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x6e]

vloxseg4ei64.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x6c]

vloxseg4ei64.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg4ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x6e]

vlseg5e8.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg5e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x80]

vlseg5e8.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg5e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x82]

vlseg5e16.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg5e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x80]

vlseg5e16.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg5e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x82]

vlseg5e32.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg5e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x80]

vlseg5e32.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg5e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x82]

vlseg5e64.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg5e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x80]

vlseg5e64.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg5e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x82]

vlseg5e8ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg5e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x81]

vlseg5e8ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg5e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0x83]

vlseg5e16ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg5e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0x81]

vlseg5e16ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg5e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x83]

vlseg5e32ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg5e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0x81]

vlseg5e32ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg5e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0x83]

vlseg5e64ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg5e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0x81]

vlseg5e64ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg5e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0x83]

vlsseg5e8.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg5e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0x88]

vlsseg5e8.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg5e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x8a]

vlsseg5e16.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg5e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0x88]

vlsseg5e16.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg5e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0x8a]

vlsseg5e32.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg5e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0x88]

vlsseg5e32.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg5e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0x8a]

vlsseg5e64.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg5e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x88]

vlsseg5e64.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg5e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x8a]

vluxseg5ei8.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg5ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x84]

vluxseg5ei8.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg5ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x86]

vluxseg5ei16.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg5ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x84]

vluxseg5ei16.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg5ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x86]

vluxseg5ei32.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg5ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x84]

vluxseg5ei32.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg5ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x86]

vluxseg5ei64.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg5ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x84]

vluxseg5ei64.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg5ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x86]

vloxseg5ei8.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg5ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0x8c]

vloxseg5ei8.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg5ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0x8e]

vloxseg5ei16.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg5ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0x8c]

vloxseg5ei16.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg5ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x8e]

vloxseg5ei32.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg5ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0x8c]

vloxseg5ei32.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg5ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0x8e]

vloxseg5ei64.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg5ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x8c]

vloxseg5ei64.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg5ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0x8e]

vlseg6e8.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg6e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xa0]

vlseg6e8.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg6e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xa2]

vlseg6e16.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg6e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xa0]

vlseg6e16.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg6e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xa2]

vlseg6e32.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg6e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xa0]

vlseg6e32.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg6e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xa2]

vlseg6e64.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg6e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xa0]

vlseg6e64.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg6e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xa2]

vlseg6e8ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg6e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xa1]

vlseg6e8ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg6e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xa3]

vlseg6e16ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg6e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xa1]

vlseg6e16ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg6e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xa3]

vlseg6e32ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg6e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xa1]

vlseg6e32ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg6e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xa3]

vlseg6e64ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg6e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xa1]

vlseg6e64ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg6e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xa3]

vlsseg6e8.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg6e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xa8]

vlsseg6e8.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg6e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xaa]

vlsseg6e16.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg6e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xa8]

vlsseg6e16.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg6e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xaa]

vlsseg6e32.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg6e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xa8]

vlsseg6e32.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg6e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xaa]

vlsseg6e64.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg6e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xa8]

vlsseg6e64.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg6e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xaa]

vluxseg6ei8.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg6ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xa4]

vluxseg6ei8.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg6ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xa6]

vluxseg6ei16.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg6ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xa4]

vluxseg6ei16.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg6ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xa6]

vluxseg6ei32.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg6ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xa4]

vluxseg6ei32.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg6ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xa6]

vluxseg6ei64.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg6ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xa4]

vluxseg6ei64.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg6ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xa6]

vloxseg6ei8.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg6ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xac]

vloxseg6ei8.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg6ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xae]

vloxseg6ei16.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg6ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xac]

vloxseg6ei16.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg6ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xae]

vloxseg6ei32.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg6ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xac]

vloxseg6ei32.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg6ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xae]

vloxseg6ei64.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg6ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xac]

vloxseg6ei64.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg6ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xae]

vlseg7e8.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg7e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xc0]

vlseg7e8.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg7e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xc2]

vlseg7e16.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg7e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xc0]

vlseg7e16.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg7e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xc2]

vlseg7e32.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg7e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xc0]

vlseg7e32.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg7e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xc2]

vlseg7e64.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg7e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xc0]

vlseg7e64.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg7e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xc2]

vlseg7e8ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg7e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xc1]

vlseg7e8ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg7e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xc3]

vlseg7e16ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg7e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xc1]

vlseg7e16ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg7e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xc3]

vlseg7e32ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg7e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xc1]

vlseg7e32ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg7e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xc3]

vlseg7e64ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg7e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xc1]

vlseg7e64ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg7e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xc3]

vlsseg7e8.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg7e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xc8]

vlsseg7e8.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg7e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xca]

vlsseg7e16.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg7e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xc8]

vlsseg7e16.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg7e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xca]

vlsseg7e32.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg7e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xc8]

vlsseg7e32.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg7e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xca]

vlsseg7e64.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg7e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xc8]

vlsseg7e64.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg7e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xca]

vluxseg7ei8.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg7ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xc4]

vluxseg7ei8.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg7ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xc6]

vluxseg7ei16.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg7ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xc4]

vluxseg7ei16.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg7ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xc6]

vluxseg7ei32.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg7ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xc4]

vluxseg7ei32.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg7ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xc6]

vluxseg7ei64.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg7ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xc4]

vluxseg7ei64.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg7ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xc6]

vloxseg7ei8.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg7ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xcc]

vloxseg7ei8.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg7ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xce]

vloxseg7ei16.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg7ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xcc]

vloxseg7ei16.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg7ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xce]

vloxseg7ei32.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg7ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xcc]

vloxseg7ei32.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg7ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xce]

vloxseg7ei64.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg7ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xcc]

vloxseg7ei64.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg7ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xce]

vlseg8e8.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg8e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xe0]

vlseg8e8.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg8e8.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xe2]

vlseg8e16.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg8e16.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xe0]

vlseg8e16.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg8e16.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xe2]

vlseg8e32.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg8e32.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xe0]

vlseg8e32.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg8e32.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xe2]

vlseg8e64.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg8e64.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xe0]

vlseg8e64.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg8e64.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xe2]

vlseg8e8ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg8e8ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0xe1]

vlseg8e8ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg8e8ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x04,0x05,0xe3]

vlseg8e16ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg8e16ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x54,0x05,0xe1]

vlseg8e16ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg8e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0xe3]

vlseg8e32ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg8e32ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x64,0x05,0xe1]

vlseg8e32ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg8e32ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x64,0x05,0xe3]

vlseg8e64ff.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg8e64ff.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xe1]

vlseg8e64ff.v v8, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg8e64ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x74,0x05,0xe3]

vlsseg8e8.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg8e8.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x04,0xb5,0xe8]

vlsseg8e8.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg8e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0xea]

vlsseg8e16.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg8e16.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x54,0xb5,0xe8]

vlsseg8e16.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg8e16.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x54,0xb5,0xea]

vlsseg8e32.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg8e32.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x64,0xb5,0xe8]

vlsseg8e32.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg8e32.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x64,0xb5,0xea]

vlsseg8e64.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg8e64.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0xe8]

vlsseg8e64.v v8, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg8e64.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0xea]

vluxseg8ei8.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg8ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xe4]

vluxseg8ei8.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg8ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xe6]

vluxseg8ei16.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg8ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xe4]

vluxseg8ei16.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg8ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xe6]

vluxseg8ei32.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg8ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xe4]

vluxseg8ei32.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg8ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xe6]

vluxseg8ei64.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg8ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xe4]

vluxseg8ei64.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg8ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xe6]

vloxseg8ei8.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg8ei8.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x04,0x45,0xec]

vloxseg8ei8.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg8ei8.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x04,0x45,0xee]

vloxseg8ei16.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg8ei16.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x54,0x45,0xec]

vloxseg8ei16.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg8ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0xee]

vloxseg8ei32.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg8ei32.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x64,0x45,0xec]

vloxseg8ei32.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg8ei32.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x64,0x45,0xee]

vloxseg8ei64.v v8, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg8ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0xec]

vloxseg8ei64.v v8, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg8ei64.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x74,0x45,0xee]

vsseg2e8.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg2e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x20]

vsseg2e8.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg2e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x22]

vsseg2e16.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg2e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x20]

vsseg2e16.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg2e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x22]

vsseg2e32.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg2e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x20]

vsseg2e32.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg2e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x22]

vsseg2e64.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg2e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x20]

vsseg2e64.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg2e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x22]

vssseg2e8.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg2e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x28]

vssseg2e8.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg2e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x2a]

vssseg2e16.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg2e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x28]

vssseg2e16.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg2e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x2a]

vssseg2e32.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg2e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x28]

vssseg2e32.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg2e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x2a]

vssseg2e64.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg2e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x28]

vssseg2e64.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg2e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x2a]

vsuxseg2ei8.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg2ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x24]

vsuxseg2ei8.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg2ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x26]

vsuxseg2ei16.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg2ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x24]

vsuxseg2ei16.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg2ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x26]

vsuxseg2ei32.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg2ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x24]

vsuxseg2ei32.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg2ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x26]

vsuxseg2ei64.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg2ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x24]

vsuxseg2ei64.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg2ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x26]

vsoxseg2ei8.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg2ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x2c]

vsoxseg2ei8.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg2ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x2e]

vsoxseg2ei16.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg2ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x2c]

vsoxseg2ei16.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg2ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x2e]

vsoxseg2ei32.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg2ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x2c]

vsoxseg2ei32.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg2ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x2e]

vsoxseg2ei64.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg2ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x2c]

vsoxseg2ei64.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg2ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x2e]

vsseg3e8.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg3e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x40]

vsseg3e8.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg3e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x42]

vsseg3e16.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg3e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x40]

vsseg3e16.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg3e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x42]

vsseg3e32.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg3e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x40]

vsseg3e32.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg3e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x42]

vsseg3e64.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg3e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x40]

vsseg3e64.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg3e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x42]

vssseg3e8.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg3e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x48]

vssseg3e8.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg3e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x4a]

vssseg3e16.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg3e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x48]

vssseg3e16.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg3e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x4a]

vssseg3e32.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg3e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x48]

vssseg3e32.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg3e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x4a]

vssseg3e64.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg3e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x48]

vssseg3e64.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg3e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x4a]

vsuxseg3ei8.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg3ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x44]

vsuxseg3ei8.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg3ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x46]

vsuxseg3ei16.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg3ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x44]

vsuxseg3ei16.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg3ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x46]

vsuxseg3ei32.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg3ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x44]

vsuxseg3ei32.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg3ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x46]

vsuxseg3ei64.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg3ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x44]

vsuxseg3ei64.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg3ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x46]

vsoxseg3ei8.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg3ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x4c]

vsoxseg3ei8.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg3ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x4e]

vsoxseg3ei16.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg3ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x4c]

vsoxseg3ei16.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg3ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x4e]

vsoxseg3ei32.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg3ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x4c]

vsoxseg3ei32.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg3ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x4e]

vsoxseg3ei64.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg3ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x4c]

vsoxseg3ei64.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg3ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x4e]

vsseg4e8.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg4e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x60]

vsseg4e8.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg4e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x62]

vsseg4e16.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg4e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x60]

vsseg4e16.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg4e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x62]

vsseg4e32.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg4e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x60]

vsseg4e32.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg4e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x62]

vsseg4e64.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg4e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x60]

vsseg4e64.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg4e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x62]

vssseg4e8.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg4e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x68]

vssseg4e8.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg4e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x6a]

vssseg4e16.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg4e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x68]

vssseg4e16.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg4e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x6a]

vssseg4e32.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg4e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x68]

vssseg4e32.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg4e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x6a]

vssseg4e64.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg4e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x68]

vssseg4e64.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg4e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x6a]

vsuxseg4ei8.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg4ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x64]

vsuxseg4ei8.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg4ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x66]

vsuxseg4ei16.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg4ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x64]

vsuxseg4ei16.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg4ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x66]

vsuxseg4ei32.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg4ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x64]

vsuxseg4ei32.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg4ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x66]

vsuxseg4ei64.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg4ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x64]

vsuxseg4ei64.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg4ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x66]

vsoxseg4ei8.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg4ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x6c]

vsoxseg4ei8.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg4ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x6e]

vsoxseg4ei16.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg4ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x6c]

vsoxseg4ei16.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg4ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x6e]

vsoxseg4ei32.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg4ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x6c]

vsoxseg4ei32.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg4ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x6e]

vsoxseg4ei64.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg4ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x6c]

vsoxseg4ei64.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg4ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x6e]

vsseg5e8.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg5e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0x80]

vsseg5e8.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg5e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0x82]

vsseg5e16.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg5e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0x80]

vsseg5e16.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg5e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0x82]

vsseg5e32.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg5e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x80]

vsseg5e32.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg5e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0x82]

vsseg5e64.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg5e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0x80]

vsseg5e64.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg5e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0x82]

vssseg5e8.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg5e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x88]

vssseg5e8.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg5e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x8a]

vssseg5e16.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg5e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x88]

vssseg5e16.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg5e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0x8a]

vssseg5e32.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg5e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x88]

vssseg5e32.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg5e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0x8a]

vssseg5e64.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg5e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x88]

vssseg5e64.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg5e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0x8a]

vsuxseg5ei8.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg5ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x84]

vsuxseg5ei8.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg5ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x86]

vsuxseg5ei16.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg5ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x84]

vsuxseg5ei16.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg5ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x86]

vsuxseg5ei32.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg5ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x84]

vsuxseg5ei32.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg5ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x86]

vsuxseg5ei64.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg5ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x84]

vsuxseg5ei64.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg5ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x86]

vsoxseg5ei8.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg5ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0x8c]

vsoxseg5ei8.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg5ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0x8e]

vsoxseg5ei16.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg5ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0x8c]

vsoxseg5ei16.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg5ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0x8e]

vsoxseg5ei32.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg5ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0x8c]

vsoxseg5ei32.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg5ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0x8e]

vsoxseg5ei64.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg5ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0x8c]

vsoxseg5ei64.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg5ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0x8e]

vsseg6e8.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg6e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0xa0]

vsseg6e8.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg6e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0xa2]

vsseg6e16.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg6e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0xa0]

vsseg6e16.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg6e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0xa2]

vsseg6e32.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg6e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0xa0]

vsseg6e32.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg6e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0xa2]

vsseg6e64.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg6e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0xa0]

vsseg6e64.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg6e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0xa2]

vssseg6e8.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg6e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xa8]

vssseg6e8.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg6e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xaa]

vssseg6e16.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg6e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xa8]

vssseg6e16.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg6e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xaa]

vssseg6e32.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg6e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xa8]

vssseg6e32.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg6e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xaa]

vssseg6e64.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg6e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xa8]

vssseg6e64.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg6e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xaa]

vsuxseg6ei8.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg6ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xa4]

vsuxseg6ei8.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg6ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xa6]

vsuxseg6ei16.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg6ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xa4]

vsuxseg6ei16.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg6ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xa6]

vsuxseg6ei32.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg6ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xa4]

vsuxseg6ei32.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg6ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xa6]

vsuxseg6ei64.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg6ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xa4]

vsuxseg6ei64.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg6ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xa6]

vsoxseg6ei8.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg6ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xac]

vsoxseg6ei8.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg6ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xae]

vsoxseg6ei16.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg6ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xac]

vsoxseg6ei16.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg6ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xae]

vsoxseg6ei32.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg6ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xac]

vsoxseg6ei32.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg6ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xae]

vsoxseg6ei64.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg6ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xac]

vsoxseg6ei64.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg6ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xae]

vsseg7e8.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg7e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0xc0]

vsseg7e8.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg7e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0xc2]

vsseg7e16.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg7e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0xc0]

vsseg7e16.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg7e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0xc2]

vsseg7e32.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg7e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0xc0]

vsseg7e32.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg7e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0xc2]

vsseg7e64.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg7e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0xc0]

vsseg7e64.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg7e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0xc2]

vssseg7e8.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg7e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xc8]

vssseg7e8.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg7e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xca]

vssseg7e16.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg7e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xc8]

vssseg7e16.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg7e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xca]

vssseg7e32.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg7e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xc8]

vssseg7e32.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg7e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xca]

vssseg7e64.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg7e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xc8]

vssseg7e64.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg7e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xca]

vsuxseg7ei8.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg7ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xc4]

vsuxseg7ei8.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg7ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xc6]

vsuxseg7ei16.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg7ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xc4]

vsuxseg7ei16.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg7ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xc6]

vsuxseg7ei32.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg7ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xc4]

vsuxseg7ei32.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg7ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xc6]

vsuxseg7ei64.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg7ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xc4]

vsuxseg7ei64.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg7ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xc6]

vsoxseg7ei8.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg7ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xcc]

vsoxseg7ei8.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg7ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xce]

vsoxseg7ei16.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg7ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xcc]

vsoxseg7ei16.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg7ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xce]

vsoxseg7ei32.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg7ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xcc]

vsoxseg7ei32.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg7ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xce]

vsoxseg7ei64.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg7ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xcc]

vsoxseg7ei64.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg7ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xce]

vsseg8e8.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg8e8.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x0c,0x05,0xe0]

vsseg8e8.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg8e8.v v24, (a0)
# CHECK-ENCODING: [0x27,0x0c,0x05,0xe2]

vsseg8e16.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg8e16.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x5c,0x05,0xe0]

vsseg8e16.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg8e16.v v24, (a0)
# CHECK-ENCODING: [0x27,0x5c,0x05,0xe2]

vsseg8e32.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg8e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0xe0]

vsseg8e32.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg8e32.v v24, (a0)
# CHECK-ENCODING: [0x27,0x6c,0x05,0xe2]

vsseg8e64.v v24, (a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg8e64.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x7c,0x05,0xe0]

vsseg8e64.v v24, (a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg8e64.v v24, (a0)
# CHECK-ENCODING: [0x27,0x7c,0x05,0xe2]

vssseg8e8.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg8e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xe8]

vssseg8e8.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg8e8.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x0c,0xb5,0xea]

vssseg8e16.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg8e16.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xe8]

vssseg8e16.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg8e16.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x5c,0xb5,0xea]

vssseg8e32.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg8e32.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xe8]

vssseg8e32.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg8e32.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x6c,0xb5,0xea]

vssseg8e64.v v24, (a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg8e64.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xe8]

vssseg8e64.v v24, (a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg8e64.v v24, (a0), a1
# CHECK-ENCODING: [0x27,0x7c,0xb5,0xea]

vsuxseg8ei8.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg8ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xe4]

vsuxseg8ei8.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg8ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xe6]

vsuxseg8ei16.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg8ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xe4]

vsuxseg8ei16.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg8ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xe6]

vsuxseg8ei32.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg8ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xe4]

vsuxseg8ei32.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg8ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xe6]

vsuxseg8ei64.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg8ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xe4]

vsuxseg8ei64.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg8ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xe6]

vsoxseg8ei8.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg8ei8.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x0c,0x45,0xec]

vsoxseg8ei8.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg8ei8.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x0c,0x45,0xee]

vsoxseg8ei16.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg8ei16.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x5c,0x45,0xec]

vsoxseg8ei16.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg8ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xee]

vsoxseg8ei32.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg8ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xec]

vsoxseg8ei32.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg8ei32.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x6c,0x45,0xee]

vsoxseg8ei64.v v24, (a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg8ei64.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x7c,0x45,0xec]

vsoxseg8ei64.v v24, (a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg8ei64.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x7c,0x45,0xee]

vlseg2e8.v v8, 0(a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e8.v v8, (a0), v0.t
# CHECK-ENCODING: [0x07,0x04,0x05,0x20]

vlseg2e16ff.v v8, 0(a0) # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlseg2e16ff.v v8, (a0)
# CHECK-ENCODING: [0x07,0x54,0x05,0x23]

vlsseg2e8.v v8, 0(a0), a1 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vlsseg2e8.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x04,0xb5,0x2a]

vluxseg3ei16.v v8, 0(a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vluxseg3ei16.v v8, (a0), v4
# CHECK-ENCODING: [0x07,0x54,0x45,0x46]

vloxseg4ei64.v v8, 0(a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors) or 'Zve64x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vloxseg4ei64.v v8, (a0), v4, v0.t
# CHECK-ENCODING: [0x07,0x74,0x45,0x6c]

vsseg5e32.v v24, 0(a0), v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsseg5e32.v v24, (a0), v0.t
# CHECK-ENCODING: [0x27,0x6c,0x05,0x80]

vssseg2e8.v v24, 0(a0), a1, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vssseg2e8.v v24, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x0c,0xb5,0x28]

vsoxseg7ei16.v v24, 0(a0), v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsoxseg7ei16.v v24, (a0), v4
# CHECK-ENCODING: [0x27,0x5c,0x45,0xce]

vsuxseg6ei32.v v24, 0(a0), v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsuxseg6ei32.v v24, (a0), v4, v0.t
# CHECK-ENCODING: [0x27,0x6c,0x45,0xa4]
