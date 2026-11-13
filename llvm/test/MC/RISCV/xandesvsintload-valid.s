# XAndesVSIntLoad - Andes Vector INT4 Load Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xandesvsintload -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+xandesvsintload < %s \
# RUN:     | llvm-objdump --mattr=+xandesvsintload -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xandesvsintload -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+xandesvsintload < %s \
# RUN:     | llvm-objdump --mattr=+xandesvsintload -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s

# CHECK-OBJ: nds.vln8.v v8, (a0)
# CHECK-ASM: nds.vln8.v v8, (a0)
# CHECK-ASM: encoding: [0x5b,0x44,0x25,0x06]
nds.vln8.v v8, (a0) # expected-error {{instruction requires the following: 'XAndesVSIntLoad' (Andes Vector INT4 Load Extension)}}

# CHECK-OBJ: nds.vln8.v v8, (a0), v0.t
# CHECK-ASM: nds.vln8.v v8, (a0), v0.t
# CHECK-ASM: encoding: [0x5b,0x44,0x25,0x04]
nds.vln8.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'XAndesVSIntLoad' (Andes Vector INT4 Load Extension)}}

# CHECK-OBJ: nds.vlnu8.v v8, (a0)
# CHECK-ASM: nds.vlnu8.v v8, (a0)
# CHECK-ASM: encoding: [0x5b,0x44,0x35,0x06]
nds.vlnu8.v v8, (a0) # expected-error {{instruction requires the following: 'XAndesVSIntLoad' (Andes Vector INT4 Load Extension)}}

# CHECK-OBJ: nds.vlnu8.v v8, (a0), v0.t
# CHECK-ASM: nds.vlnu8.v v8, (a0), v0.t
# CHECK-ASM: encoding: [0x5b,0x44,0x35,0x04]
nds.vlnu8.v v8, (a0), v0.t # expected-error {{instruction requires the following: 'XAndesVSIntLoad' (Andes Vector INT4 Load Extension)}}
