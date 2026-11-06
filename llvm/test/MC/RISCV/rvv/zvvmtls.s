# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvvmtls %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvvmtls %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvvmtls - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vmtl.v v8, (a0), a1
# CHECK-INST: vmtl.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x12]
# expected-error@-3 {{instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store)}}

vmtl.v v8, (a0), a1, L4
# CHECK-INST: vmtl.v v8, (a0), a1, L4
# CHECK-ENCODING: [0x07,0x74,0xb5,0x72]
# expected-error@-3 {{instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store)}}

vmtl.v v8, (a0), a1, l2
# CHECK-INST: vmtl.v v8, (a0), a1, L2
# CHECK-ENCODING: [0x07,0x74,0xb5,0x52]
# expected-error@-3 {{instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store)}}

vmtl.v v8, (a0), a1, v0.t
# CHECK-INST: vmtl.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x10]
# expected-error@-3 {{instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store)}}

vmtl.v v8, (a0), a1, L4, v0.t
# CHECK-INST: vmtl.v v8, (a0), a1, L4, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x70]
# expected-error@-3 {{instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store)}}

vmtl.v v8, (a0), zero, L64, v0.t
# CHECK-INST: vmtl.v v8, (a0), zero, L64, v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xf0]
# expected-error@-3 {{instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store)}}

vmts.v v12, (a0), a1
# CHECK-INST: vmts.v v12, (a0), a1
# CHECK-ENCODING: [0x27,0x76,0xb5,0x12]
# expected-error@-3 {{instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store)}}

vmts.v v12, (a0), a1, L4
# CHECK-INST: vmts.v v12, (a0), a1, L4
# CHECK-ENCODING: [0x27,0x76,0xb5,0x72]
# expected-error@-3 {{instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store)}}

vmts.v v12, (a0), a1, v0.t
# CHECK-INST: vmts.v v12, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x76,0xb5,0x10]
# expected-error@-3 {{instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store)}}

vmts.v v12, (a0), a1, L4, v0.t
# CHECK-INST: vmts.v v12, (a0), a1, L4, v0.t
# CHECK-ENCODING: [0x27,0x76,0xb5,0x70]
# expected-error@-3 {{instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store)}}

vmts.v v12, (a0), a1, l2, v0.t
# CHECK-INST: vmts.v v12, (a0), a1, L2, v0.t
# CHECK-ENCODING: [0x27,0x76,0xb5,0x50]
# expected-error@-3 {{instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store)}}

vmts.v v12, (a0), zero, L64, v0.t
# CHECK-INST: vmts.v v12, (a0), zero, L64, v0.t
# CHECK-ENCODING: [0x27,0x76,0x05,0xf0]
# expected-error@-3 {{instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store)}}
