# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvdot4a8i %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvdot4a8i %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvdot4a8i - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vdot4a.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers)}}
# CHECK-INST: vdot4a.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xb0]

vdot4a.vv v8, v4, v20 # expected-error {{instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers)}}
# CHECK-INST: vdot4a.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xb2]

vdot4au.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers)}}
# CHECK-INST: vdot4au.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa0]

vdot4au.vv v8, v4, v20 # expected-error {{instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers)}}
# CHECK-INST: vdot4au.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa2]

vdot4asu.vv v8, v4, v20, v0.t # expected-error {{instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers)}}
# CHECK-INST: vdot4asu.vv v8, v4, v20, v0.t
# CHECK-ENCODING: [0x57,0x24,0x4a,0xa8]

vdot4asu.vv v8, v4, v20 # expected-error {{instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers)}}
# CHECK-INST: vdot4asu.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x24,0x4a,0xaa]

vdot4a.vx v8, v4, s4, v0.t # expected-error {{instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers)}}
# CHECK-INST: vdot4a.vx v8, v4, s4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x4a,0xb0]

vdot4a.vx v8, v4, s4 # expected-error {{instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers)}}
# CHECK-INST: vdot4a.vx v8, v4, s4
# CHECK-ENCODING: [0x57,0x64,0x4a,0xb2]

vdot4au.vx v8, v4, s4, v0.t # expected-error {{instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers)}}
# CHECK-INST: vdot4au.vx v8, v4, s4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x4a,0xa0]

vdot4au.vx v8, v4, s4 # expected-error {{instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers)}}
# CHECK-INST: vdot4au.vx v8, v4, s4
# CHECK-ENCODING: [0x57,0x64,0x4a,0xa2]

vdot4asu.vx v8, v4, s4, v0.t # expected-error {{instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers)}}
# CHECK-INST: vdot4asu.vx v8, v4, s4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x4a,0xa8]

vdot4asu.vx v8, v4, s4 # expected-error {{instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers)}}
# CHECK-INST: vdot4asu.vx v8, v4, s4
# CHECK-ENCODING: [0x57,0x64,0x4a,0xaa]

vdot4aus.vx v8, v4, s4, v0.t # expected-error {{instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers)}}
# CHECK-INST: vdot4aus.vx v8, v4, s4, v0.t
# CHECK-ENCODING: [0x57,0x64,0x4a,0xb8]

vdot4aus.vx v8, v4, s4 # expected-error {{instruction requires the following: 'Zvdot4a8i' (Vector 4-element Dot Product of packed 8-bit Integers)}}
# CHECK-INST: vdot4aus.vx v8, v4, s4
# CHECK-ENCODING: [0x57,0x64,0x4a,0xba]
