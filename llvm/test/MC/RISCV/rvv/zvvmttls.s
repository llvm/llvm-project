# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvvmttls %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvvmttls %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvvmttls - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vmttl.v v8, (a0), a1
# CHECK-INST: vmttl.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x16]
# CHECK-ERROR: instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store){{$}}

vmttl.v v8, (a0), a1, L4
# CHECK-INST: vmttl.v v8, (a0), a1, L4
# CHECK-ENCODING: [0x07,0x74,0xb5,0x76]
# CHECK-ERROR: instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store){{$}}

vmttl.v v8, (a0), a1, l2
# CHECK-INST: vmttl.v v8, (a0), a1, L2
# CHECK-ENCODING: [0x07,0x74,0xb5,0x56]
# CHECK-ERROR: instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store){{$}}

vmttl.v v8, (a0), a1, v0.t
# CHECK-INST: vmttl.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x14]
# CHECK-ERROR: instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store){{$}}

vmttl.v v8, (a0), a1, L4, v0.t
# CHECK-INST: vmttl.v v8, (a0), a1, L4, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x74]
# CHECK-ERROR: instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store){{$}}

vmttl.v v8, (a0), zero, L64, v0.t
# CHECK-INST: vmttl.v v8, (a0), zero, L64, v0.t
# CHECK-ENCODING: [0x07,0x74,0x05,0xf4]
# CHECK-ERROR: instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store){{$}}

vmtts.v v12, (a0), a1
# CHECK-INST: vmtts.v v12, (a0), a1
# CHECK-ENCODING: [0x27,0x76,0xb5,0x16]
# CHECK-ERROR: instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store){{$}}

vmtts.v v12, (a0), a1, L4
# CHECK-INST: vmtts.v v12, (a0), a1, L4
# CHECK-ENCODING: [0x27,0x76,0xb5,0x76]
# CHECK-ERROR: instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store){{$}}

vmtts.v v12, (a0), a1, v0.t
# CHECK-INST: vmtts.v v12, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x76,0xb5,0x14]
# CHECK-ERROR: instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store){{$}}

vmtts.v v12, (a0), a1, L4, v0.t
# CHECK-INST: vmtts.v v12, (a0), a1, L4, v0.t
# CHECK-ENCODING: [0x27,0x76,0xb5,0x74]
# CHECK-ERROR: instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store){{$}}

vmtts.v v12, (a0), a1, l2, v0.t
# CHECK-INST: vmtts.v v12, (a0), a1, L2, v0.t
# CHECK-ENCODING: [0x27,0x76,0xb5,0x54]
# CHECK-ERROR: instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store){{$}}

vmtts.v v12, (a0), zero, L64, v0.t
# CHECK-INST: vmtts.v v12, (a0), zero, L64, v0.t
# CHECK-ENCODING: [0x27,0x76,0x05,0xf4]
# CHECK-ERROR: instruction requires the following: 'Zvvmttls' (Transposing Matrix Tile Load/Store){{$}}
