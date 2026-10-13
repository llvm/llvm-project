# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v %s \
# RUN:        | llvm-objdump -d --mattr=+v - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vmul.vv v8, v4, v20, v0.t
# CHECK-INST: vmul.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x94]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmul.vv v8, v4, v20
# CHECK-INST: vmul.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x96]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmul.vx v8, v4, a0, v0.t
# CHECK-INST: vmul.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x94]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmul.vx v8, v4, a0
# CHECK-INST: vmul.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x96]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmulh.vv v8, v4, v20, v0.t
# CHECK-INST: vmulh.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x9c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmulh.vv v8, v4, v20
# CHECK-INST: vmulh.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x9e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmulh.vx v8, v4, a0, v0.t
# CHECK-INST: vmulh.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x9c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmulh.vx v8, v4, a0
# CHECK-INST: vmulh.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x9e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmulhu.vv v8, v4, v20, v0.t
# CHECK-INST: vmulhu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x90]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmulhu.vv v8, v4, v20
# CHECK-INST: vmulhu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x92]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmulhu.vx v8, v4, a0, v0.t
# CHECK-INST: vmulhu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x90]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmulhu.vx v8, v4, a0
# CHECK-INST: vmulhu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x92]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmulhsu.vv v8, v4, v20, v0.t
# CHECK-INST: vmulhsu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0x98]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmulhsu.vv v8, v4, v20
# CHECK-INST: vmulhsu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0x9a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmulhsu.vx v8, v4, a0, v0.t
# CHECK-INST: vmulhsu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0x98]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vmulhsu.vx v8, v4, a0
# CHECK-INST: vmulhsu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0x9a]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmul.vv v8, v4, v20, v0.t
# CHECK-INST: vwmul.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xec]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmul.vv v8, v4, v20
# CHECK-INST: vwmul.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xee]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmul.vx v8, v4, a0, v0.t
# CHECK-INST: vwmul.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xec]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmul.vx v8, v4, a0
# CHECK-INST: vwmul.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xee]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmulu.vv v8, v4, v20, v0.t
# CHECK-INST: vwmulu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xe0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmulu.vv v8, v4, v20
# CHECK-INST: vwmulu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmulu.vx v8, v4, a0, v0.t
# CHECK-INST: vwmulu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xe0]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmulu.vx v8, v4, a0
# CHECK-INST: vwmulu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xe2]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmulsu.vv v8, v4, v20, v0.t
# CHECK-INST: vwmulsu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xe8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmulsu.vv v8, v4, v20
# CHECK-INST: vwmulsu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xea]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmulsu.vx v8, v4, a0, v0.t
# CHECK-INST: vwmulsu.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x64,0x45,0xe8]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vwmulsu.vx v8, v4, a0
# CHECK-INST: vwmulsu.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x64,0x45,0xea]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsmul.vv v8, v4, v20, v0.t
# CHECK-INST: vsmul.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x04,0x4a,0x9c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsmul.vv v8, v4, v20
# CHECK-INST: vsmul.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x4a,0x9e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsmul.vx v8, v4, a0, v0.t
# CHECK-INST: vsmul.vx v8, v4, a0, v0.t
# CHECK-ENCODING: [0x57,0x44,0x45,0x9c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vsmul.vx v8, v4, a0
# CHECK-INST: vsmul.vx v8, v4, a0
# CHECK-ENCODING: [0x57,0x44,0x45,0x9e]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}
