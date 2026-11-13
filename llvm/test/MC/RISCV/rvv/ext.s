# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vzext.vf2 v8, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vzext.vf2 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x43,0x48]

vzext.vf2 v8, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vzext.vf2 v8, v4
# CHECK-ENCODING: [0x57,0x24,0x43,0x4a]

vsext.vf2 v8, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsext.vf2 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x43,0x48]

vsext.vf2 v8, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsext.vf2 v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x43,0x4a]

vzext.vf4 v8, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vzext.vf4 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x42,0x48]

vzext.vf4 v8, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vzext.vf4 v8, v4
# CHECK-ENCODING: [0x57,0x24,0x42,0x4a]

vsext.vf4 v8, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsext.vf4 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x42,0x48]

vsext.vf4 v8, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsext.vf4 v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x42,0x4a]

vzext.vf8 v8, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vzext.vf8 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x41,0x48]

vzext.vf8 v8, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vzext.vf8 v8, v4
# CHECK-ENCODING: [0x57,0x24,0x41,0x4a]

vsext.vf8 v8, v4, v0.t # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsext.vf8 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x41,0x48]

vsext.vf8 v8, v4 # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsext.vf8 v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x41,0x4a]
