# XAndesVSIntLoad - Andes Vector INT4 Load Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+xandesvsintload -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+xandesvsintload < %s \
# RUN:     | llvm-objdump --mattr=+xandesvsintload -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc %s -triple=riscv64 -mattr=+xandesvsintload -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+xandesvsintload < %s \
# RUN:     | llvm-objdump --mattr=+xandesvsintload -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ %s
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR

# CHECK-OBJ: nds.vln8.v v8, (a0)
# CHECK-ASM: nds.vln8.v v8, (a0)
# CHECK-ASM: encoding: [0x5b,0x44,0x25,0x06]
# CHECK-ERROR: instruction requires the following: 'XAndesVSIntLoad' (Andes Vector INT4 Load Extension){{$}}
nds.vln8.v v8, (a0)

# CHECK-OBJ: nds.vln8.v v8, (a0), v0.t
# CHECK-ASM: nds.vln8.v v8, (a0), v0.t
# CHECK-ASM: encoding: [0x5b,0x44,0x25,0x04]
# CHECK-ERROR: instruction requires the following: 'XAndesVSIntLoad' (Andes Vector INT4 Load Extension){{$}}
nds.vln8.v v8, (a0), v0.t

# CHECK-OBJ: nds.vlnu8.v v8, (a0)
# CHECK-ASM: nds.vlnu8.v v8, (a0)
# CHECK-ASM: encoding: [0x5b,0x44,0x35,0x06]
# CHECK-ERROR: instruction requires the following: 'XAndesVSIntLoad' (Andes Vector INT4 Load Extension){{$}}
nds.vlnu8.v v8, (a0)

# CHECK-OBJ: nds.vlnu8.v v8, (a0), v0.t
# CHECK-ASM: nds.vlnu8.v v8, (a0), v0.t
# CHECK-ASM: encoding: [0x5b,0x44,0x35,0x04]
# CHECK-ERROR: instruction requires the following: 'XAndesVSIntLoad' (Andes Vector INT4 Load Extension){{$}}
nds.vlnu8.v v8, (a0), v0.t
