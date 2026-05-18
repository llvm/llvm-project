# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvvmtls %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvvmtls %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvvmtls - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vmtl.v v8, (a0), a1
# CHECK-INST: vmtl.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x12]
# CHECK-ERROR: instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store){{$}}

vmtl.v v8, (a0), a1, L4
# CHECK-INST: vmtl.v v8, (a0), a1, L4
# CHECK-ENCODING: [0x07,0x74,0xb5,0x72]
# CHECK-ERROR: instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store){{$}}

vmtl.v v8, (a0), a1, v0.t
# CHECK-INST: vmtl.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x10]
# CHECK-ERROR: instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store){{$}}

vmtl.v v8, (a0), a1, L4, v0.t
# CHECK-INST: vmtl.v v8, (a0), a1, L4, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x70]
# CHECK-ERROR: instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store){{$}}

vmtl.v v8, (a0), zero, L64, v0.t
# CHECK-INST: vmtl.v v8, (a0), zero, L64, v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store){{$}}

vmts.v v12, (a0), a1
# CHECK-INST: vmts.v v12, (a0), a1
# CHECK-ENCODING: [0x27,0x76,0xb5,0x12]
# CHECK-ERROR: instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store){{$}}

vmts.v v12, (a0), a1, L4
# CHECK-INST: vmts.v v12, (a0), a1, L4
# CHECK-ENCODING: [0x27,0x76,0xb5,0x72]
# CHECK-ERROR: instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store){{$}}

vmts.v v12, (a0), a1, v0.t
# CHECK-INST: vmts.v v12, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x76,0xb5,0x10]
# CHECK-ERROR: instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store){{$}}

vmts.v v12, (a0), a1, L4, v0.t
# CHECK-INST: vmts.v v12, (a0), a1, L4, v0.t
# CHECK-ENCODING: [0x27,0x76,0xb5,0x70]
# CHECK-ERROR: instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store){{$}}

vmts.v v12, (a0), zero, L64, v0.t
# CHECK-INST: vmts.v v12, (a0), zero, L64, v0.t
# CHECK-ENCODING: [0x27,0x76,0x05,0xf0]
# CHECK-ERROR: instruction requires the following: 'Zvvmtls' (Matrix Tile Load/Store){{$}}
