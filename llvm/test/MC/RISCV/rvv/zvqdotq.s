# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvqdotq %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvqdotq %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvqdotq - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvqdotq %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vqdot.vv v8, v4, v20, v0.t
# CHECK-INST: vqdot.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xb0]
# CHECK-ERROR: instruction requires the following: 'Zvqdotq' (Vector quad widening 4D Dot Product){{$}}
# CHECK-UNKNOWN: b04a2457 <unknown>

vqdot.vv v8, v4, v20
# CHECK-INST: vqdot.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xb2]
# CHECK-ERROR: instruction requires the following: 'Zvqdotq' (Vector quad widening 4D Dot Product){{$}}
# CHECK-UNKNOWN: b24a2457 <unknown>

vqdotu.vv v8, v4, v20, v0.t
# CHECK-INST: vqdotu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvqdotq' (Vector quad widening 4D Dot Product){{$}}
# CHECK-UNKNOWN: a04a2457 <unknown>

vqdotu.vv v8, v4, v20
# CHECK-INST: vqdotu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvqdotq' (Vector quad widening 4D Dot Product){{$}}
# CHECK-UNKNOWN: a24a2457 <unknown>

vqdotsu.vv v8, v4, v20, v0.t
# CHECK-INST: vqdotsu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvqdotq' (Vector quad widening 4D Dot Product){{$}}
# CHECK-UNKNOWN: a84a2457 <unknown>

vqdotsu.vv v8, v4, v20
# CHECK-INST: vqdotsu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvqdotq' (Vector quad widening 4D Dot Product){{$}}
# CHECK-UNKNOWN: aa4a2457 <unknown>

vqdot.vx v8, v4, s4, v0.t
# CHECK-INST: vqdot.vx v8, v4, s4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x4a,0xb0]
# CHECK-ERROR: instruction requires the following: 'Zvqdotq' (Vector quad widening 4D Dot Product){{$}}
# CHECK-UNKNOWN: b04a6457 <unknown>

vqdot.vx v8, v4, s4
# CHECK-INST: vqdot.vx v8, v4, s4
# CHECK-ENCODING: [0x57,0x64,0x4a,0xb2]
# CHECK-ERROR: instruction requires the following: 'Zvqdotq' (Vector quad widening 4D Dot Product){{$}}
# CHECK-UNKNOWN: b24a6457 <unknown>

vqdotu.vx v8, v4, s4, v0.t
# CHECK-INST: vqdotu.vx v8, v4, s4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x4a,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvqdotq' (Vector quad widening 4D Dot Product){{$}}
# CHECK-UNKNOWN: a04a6457 <unknown>

vqdotu.vx v8, v4, s4
# CHECK-INST: vqdotu.vx v8, v4, s4
# CHECK-ENCODING: [0x57,0x64,0x4a,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvqdotq' (Vector quad widening 4D Dot Product){{$}}
# CHECK-UNKNOWN: a24a6457 <unknown>

vqdotsu.vx v8, v4, s4, v0.t
# CHECK-INST: vqdotsu.vx v8, v4, s4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x4a,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvqdotq' (Vector quad widening 4D Dot Product){{$}}
# CHECK-UNKNOWN: a84a6457 <unknown>

vqdotsu.vx v8, v4, s4
# CHECK-INST: vqdotsu.vx v8, v4, s4
# CHECK-ENCODING: [0x57,0x64,0x4a,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvqdotq' (Vector quad widening 4D Dot Product){{$}}
# CHECK-UNKNOWN: aa4a6457 <unknown>

vqdotus.vx v8, v4, s4, v0.t
# CHECK-INST: vqdotus.vx v8, v4, s4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x4a,0xb8]
# CHECK-ERROR: instruction requires the following: 'Zvqdotq' (Vector quad widening 4D Dot Product){{$}}
# CHECK-UNKNOWN: b84a6457 <unknown>

vqdotus.vx v8, v4, s4
# CHECK-INST: vqdotus.vx v8, v4, s4
# CHECK-ENCODING: [0x57,0x64,0x4a,0xba]
# CHECK-ERROR: instruction requires the following: 'Zvqdotq' (Vector quad widening 4D Dot Product){{$}}
# CHECK-UNKNOWN: ba4a6457 <unknown>
