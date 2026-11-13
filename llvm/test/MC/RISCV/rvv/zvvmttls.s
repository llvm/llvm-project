# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvvmttls %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvvmttls %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvvmttls - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vmttl.v v8, (a0), a1 # expected-error {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}
# CHECK-INST: vmttl.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x16]

vmttl.v v8, (a0), a1, L4 # expected-error {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}
# CHECK-INST: vmttl.v v8, (a0), a1, L4
# CHECK-ENCODING: [0x07,0x74,0xb5,0x76]

vmttl.v v8, (a0), a1, l2 # expected-error {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}
# CHECK-INST: vmttl.v v8, (a0), a1, L2
# CHECK-ENCODING: [0x07,0x74,0xb5,0x56]

vmttl.v v8, (a0), a1, v0.t # expected-error {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}
# CHECK-INST: vmttl.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x14]

vmttl.v v8, (a0), a1, L4, v0.t # expected-error {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}
# CHECK-INST: vmttl.v v8, (a0), a1, L4, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x74]

vmttl.v v8, (a0), zero, L64, v0.t # expected-error {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}
# CHECK-INST: vmttl.v v8, (a0), zero, L64, v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xf4]

vmtts.v v12, (a0), a1 # expected-error {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}
# CHECK-INST: vmtts.v v12, (a0), a1
# CHECK-ENCODING: [0x27,0x76,0xb5,0x16]

vmtts.v v12, (a0), a1, L4 # expected-error {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}
# CHECK-INST: vmtts.v v12, (a0), a1, L4
# CHECK-ENCODING: [0x27,0x76,0xb5,0x76]

vmtts.v v12, (a0), a1, v0.t # expected-error {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}
# CHECK-INST: vmtts.v v12, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x76,0xb5,0x14]

vmtts.v v12, (a0), a1, L4, v0.t # expected-error {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}
# CHECK-INST: vmtts.v v12, (a0), a1, L4, v0.t
# CHECK-ENCODING: [0x27,0x76,0xb5,0x74]

vmtts.v v12, (a0), a1, l2, v0.t # expected-error {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}
# CHECK-INST: vmtts.v v12, (a0), a1, L2, v0.t
# CHECK-ENCODING: [0x27,0x76,0xb5,0x54]

vmtts.v v12, (a0), zero, L64, v0.t # expected-error {{instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store)}}
# CHECK-INST: vmtts.v v12, (a0), zero, L64, v0.t
# CHECK-ENCODING: [0x27,0x76,0x05,0xf4]
