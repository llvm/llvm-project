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
# CHECK-UNKNOWN: 48432457 <unknown>

vzext.vf2 v8, v4
# CHECK-INST: vzext.vf2 v8, v4
# CHECK-ENCODING: [0x57,0x24,0x43,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a432457 <unknown>

vsext.vf2 v8, v4, v0.t
# CHECK-INST: vsext.vf2 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x43,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4843a457 <unknown>

vsext.vf2 v8, v4
# CHECK-INST: vsext.vf2 v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x43,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a43a457 <unknown>

vzext.vf4 v8, v4, v0.t
# CHECK-INST: vzext.vf4 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x42,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48422457 <unknown>

vzext.vf4 v8, v4
# CHECK-INST: vzext.vf4 v8, v4
# CHECK-ENCODING: [0x57,0x24,0x42,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a422457 <unknown>

vsext.vf4 v8, v4, v0.t
# CHECK-INST: vsext.vf4 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x42,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4842a457 <unknown>

vsext.vf4 v8, v4
# CHECK-INST: vsext.vf4 v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x42,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a42a457 <unknown>

vzext.vf8 v8, v4, v0.t
# CHECK-INST: vzext.vf8 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0x24,0x41,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 48412457 <unknown>

vzext.vf8 v8, v4
# CHECK-INST: vzext.vf8 v8, v4
# CHECK-ENCODING: [0x57,0x24,0x41,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a412457 <unknown>

vsext.vf8 v8, v4, v0.t
# CHECK-INST: vsext.vf8 v8, v4, v0.t
# CHECK-ENCODING: [0x57,0xa4,0x41,0x48]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4841a457 <unknown>

vsext.vf8 v8, v4
# CHECK-INST: vsext.vf8 v8, v4
# CHECK-ENCODING: [0x57,0xa4,0x41,0x4a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
# CHECK-UNKNOWN: 4a41a457 <unknown>
