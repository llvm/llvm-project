# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vredsum.vs v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredsum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x00]

vredsum.vs v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredsum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x02]

vredmaxu.vs v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredmaxu.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x18]

vredmaxu.vs v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredmaxu.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x1a]

vredmax.vs v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredmax.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x1c]

vredmax.vs v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredmax.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x1e]

vredminu.vs v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredminu.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x10]

vredminu.vs v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredminu.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x12]

vredmin.vs v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredmin.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x14]

vredmin.vs v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredmin.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x16]

vredand.vs v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredand.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x04]

vredand.vs v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredand.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x06]

vredor.vs v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredor.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x08]

vredor.vs v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredor.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x0a]

vredxor.vs v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredxor.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x0c]

vredxor.vs v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredxor.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x0e]

vwredsumu.vs v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwredsumu.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xc0]

vwredsumu.vs v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwredsumu.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xc2]

vwredsum.vs v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwredsum.vs v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0xc4]

vwredsum.vs v8, v4, v20 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vwredsum.vs v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0xc6]

vredsum.vs v0, v4, v20, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vredsum.vs v0, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x20,0x4a,0x00]
