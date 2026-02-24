# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvdot4a8i %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvdot4a8i %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvdot4a8i - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvdot4a8i %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vdota4.vv v8, v4, v20, v0.t
# CHECK-INST: vdota4.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xb0]
# CHECK-ERROR: instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers){{$}}
# CHECK-UNKNOWN: b04a2457 <unknown>

vdota4.vv v8, v4, v20
# CHECK-INST: vdota4.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xb2]
# CHECK-ERROR: instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers){{$}}
# CHECK-UNKNOWN: b24a2457 <unknown>

vdota4u.vv v8, v4, v20, v0.t
# CHECK-INST: vdota4u.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers){{$}}
# CHECK-UNKNOWN: a04a2457 <unknown>

vdota4u.vv v8, v4, v20
# CHECK-INST: vdota4u.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers){{$}}
# CHECK-UNKNOWN: a24a2457 <unknown>

vdota4su.vv v8, v4, v20, v0.t
# CHECK-INST: vdota4su.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers){{$}}
# CHECK-UNKNOWN: a84a2457 <unknown>

vdota4su.vv v8, v4, v20
# CHECK-INST: vdota4su.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers){{$}}
# CHECK-UNKNOWN: aa4a2457 <unknown>

vdota4.vx v8, v4, s4, v0.t
# CHECK-INST: vdota4.vx v8, v4, s4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x4a,0xb0]
# CHECK-ERROR: instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers){{$}}
# CHECK-UNKNOWN: b04a6457 <unknown>

vdota4.vx v8, v4, s4
# CHECK-INST: vdota4.vx v8, v4, s4
# CHECK-ENCODING: [0x57,0x64,0x4a,0xb2]
# CHECK-ERROR: instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers){{$}}
# CHECK-UNKNOWN: b24a6457 <unknown>

vdota4u.vx v8, v4, s4, v0.t
# CHECK-INST: vdota4u.vx v8, v4, s4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x4a,0xa0]
# CHECK-ERROR: instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers){{$}}
# CHECK-UNKNOWN: a04a6457 <unknown>

vdota4u.vx v8, v4, s4
# CHECK-INST: vdota4u.vx v8, v4, s4
# CHECK-ENCODING: [0x57,0x64,0x4a,0xa2]
# CHECK-ERROR: instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers){{$}}
# CHECK-UNKNOWN: a24a6457 <unknown>

vdota4su.vx v8, v4, s4, v0.t
# CHECK-INST: vdota4su.vx v8, v4, s4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x4a,0xa8]
# CHECK-ERROR: instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers){{$}}
# CHECK-UNKNOWN: a84a6457 <unknown>

vdota4su.vx v8, v4, s4
# CHECK-INST: vdota4su.vx v8, v4, s4
# CHECK-ENCODING: [0x57,0x64,0x4a,0xaa]
# CHECK-ERROR: instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers){{$}}
# CHECK-UNKNOWN: aa4a6457 <unknown>

vdota4us.vx v8, v4, s4, v0.t
# CHECK-INST: vdota4us.vx v8, v4, s4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x4a,0xb8]
# CHECK-ERROR: instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers){{$}}
# CHECK-UNKNOWN: b84a6457 <unknown>

vdota4us.vx v8, v4, s4
# CHECK-INST: vdota4us.vx v8, v4, s4
# CHECK-ENCODING: [0x57,0x64,0x4a,0xba]
# CHECK-ERROR: instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers){{$}}
# CHECK-UNKNOWN: ba4a6457 <unknown>
