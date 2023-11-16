# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vzext.vf2 v8, v4, v0.t
# CHECK-INST: vzext.vf2 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x43,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 24 43 48 <unknown>

vzext.vf2 v8, v4
# CHECK-INST: vzext.vf2 v8, v4
# CHECK-ENCODING: [0x57,0x24,0x43,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 24 43 4a <unknown>

vsext.vf2 v8, v4, v0.t
# CHECK-INST: vsext.vf2 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x43,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 a4 43 48 <unknown>

vsext.vf2 v8, v4
# CHECK-INST: vsext.vf2 v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x43,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 a4 43 4a <unknown>

vzext.vf4 v8, v4, v0.t
# CHECK-INST: vzext.vf4 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x42,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 24 42 48 <unknown>

vzext.vf4 v8, v4
# CHECK-INST: vzext.vf4 v8, v4
# CHECK-ENCODING: [0x57,0x24,0x42,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 24 42 4a <unknown>

vsext.vf4 v8, v4, v0.t
# CHECK-INST: vsext.vf4 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x42,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 a4 42 48 <unknown>

vsext.vf4 v8, v4
# CHECK-INST: vsext.vf4 v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x42,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 a4 42 4a <unknown>

vzext.vf8 v8, v4, v0.t
# CHECK-INST: vzext.vf8 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x41,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 24 41 48 <unknown>

vzext.vf8 v8, v4
# CHECK-INST: vzext.vf8 v8, v4
# CHECK-ENCODING: [0x57,0x24,0x41,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 24 41 4a <unknown>

vsext.vf8 v8, v4, v0.t
# CHECK-INST: vsext.vf8 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x41,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 a4 41 48 <unknown>

vsext.vf8 v8, v4
# CHECK-INST: vsext.vf8 v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x41,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 57 a4 41 4a <unknown>
