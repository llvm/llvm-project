# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvvmttls %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvvmttls %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvvmttls - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vmttl.v v8, (a0), a1
# CHECK-INST: vmttl.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x16]
# expected-error@-3 {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}

vmttl.v v8, (a0), a1, L4
# CHECK-INST: vmttl.v v8, (a0), a1, L4
# CHECK-ENCODING: [0x07,0x74,0xb5,0x76]
# expected-error@-3 {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}

vmttl.v v8, (a0), a1, l2
# CHECK-INST: vmttl.v v8, (a0), a1, L2
# CHECK-ENCODING: [0x07,0x74,0xb5,0x56]
# expected-error@-3 {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}

vmttl.v v8, (a0), a1, v0.t
# CHECK-INST: vmttl.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x14]
# expected-error@-3 {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}

vmttl.v v8, (a0), a1, L4, v0.t
# CHECK-INST: vmttl.v v8, (a0), a1, L4, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x74]
# expected-error@-3 {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}

vmttl.v v8, (a0), zero, L64, v0.t
# CHECK-INST: vmttl.v v8, (a0), zero, L64, v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xf4]
# expected-error@-3 {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}

vmtts.v v12, (a0), a1
# CHECK-INST: vmtts.v v12, (a0), a1
# CHECK-ENCODING: [0x27,0x76,0xb5,0x16]
# expected-error@-3 {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}

vmtts.v v12, (a0), a1, L4
# CHECK-INST: vmtts.v v12, (a0), a1, L4
# CHECK-ENCODING: [0x27,0x76,0xb5,0x76]
# expected-error@-3 {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}

vmtts.v v12, (a0), a1, v0.t
# CHECK-INST: vmtts.v v12, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x76,0xb5,0x14]
# expected-error@-3 {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}

vmtts.v v12, (a0), a1, L4, v0.t
# CHECK-INST: vmtts.v v12, (a0), a1, L4, v0.t
# CHECK-ENCODING: [0x27,0x76,0xb5,0x74]
# expected-error@-3 {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}

vmtts.v v12, (a0), a1, l2, v0.t
# CHECK-INST: vmtts.v v12, (a0), a1, L2, v0.t
# CHECK-ENCODING: [0x27,0x76,0xb5,0x54]
# expected-error@-3 {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}

vmtts.v v12, (a0), zero, L64, v0.t
# CHECK-INST: vmtts.v v12, (a0), zero, L64, v0.t
# CHECK-ENCODING: [0x27,0x76,0x05,0xf4]
# expected-error@-3 {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}
