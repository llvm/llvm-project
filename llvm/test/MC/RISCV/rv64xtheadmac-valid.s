# RUN: llvm-mc %s -triple=riscv64 -mattr=+xtheadmac -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+xtheadmac < %s \
# RUN:     | llvm-objdump --mattr=+xtheadmac -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: th.mula a0, a1, a2
# CHECK-ASM: encoding: [0x0b,0x95,0xc5,0x20]
th.mula	 a0, a1, a2

# CHECK-ASM-AND-OBJ: th.mulah a0, a1, a2
# CHECK-ASM: encoding: [0x0b,0x95,0xc5,0x28]
th.mulah a0, a1, a2

# CHECK-ASM-AND-OBJ: th.mulaw a0, a1, a2
# CHECK-ASM: encoding: [0x0b,0x95,0xc5,0x24]
th.mulaw a0, a1, a2

# CHECK-ASM-AND-OBJ: th.muls a0, a1, a2
# CHECK-ASM: encoding: [0x0b,0x95,0xc5,0x22]
th.muls	 a0, a1, a2

# CHECK-ASM-AND-OBJ: th.mulsh a0, a1, a2
# CHECK-ASM: encoding: [0x0b,0x95,0xc5,0x2a]
th.mulsh a0, a1, a2

# CHECK-ASM-AND-OBJ: th.mulsw a0, a1, a2
# CHECK-ASM: encoding: [0x0b,0x95,0xc5,0x26]
th.mulsw a0, a1, a2
