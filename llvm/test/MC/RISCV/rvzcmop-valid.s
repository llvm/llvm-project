# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zcmop -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zcmop -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zcmop < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zcmop -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zcmop < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zcmop -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: cmop.1
# CHECK-ASM: encoding: [0x81,0x60]
cmop.1

# CHECK-ASM-AND-OBJ: cmop.3
# CHECK-ASM: encoding: [0x81,0x61]
cmop.3

# CHECK-ASM-AND-OBJ: cmop.5
# CHECK-ASM: encoding: [0x81,0x62]
cmop.5

# CHECK-ASM-AND-OBJ: cmop.7
# CHECK-ASM: encoding: [0x81,0x63]
cmop.7

# CHECK-ASM-AND-OBJ: cmop.9
# CHECK-ASM: encoding: [0x81,0x64]
cmop.9

# CHECK-ASM-AND-OBJ: cmop.11
# CHECK-ASM: encoding: [0x81,0x65]
cmop.11

# CHECK-ASM-AND-OBJ: cmop.13
# CHECK-ASM: encoding: [0x81,0x66]
cmop.13

# CHECK-ASM-AND-OBJ: cmop.15
# CHECK-ASM: encoding: [0x81,0x67]
cmop.15
