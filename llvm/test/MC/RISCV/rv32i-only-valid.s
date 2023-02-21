# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 < %s \
# RUN:     | llvm-objdump -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: addi a0, a1, -1
# CHECK-ASM: encoding: [0x13,0x85,0xf5,0xff]
addi a0, a1, 4294967295 # 0xffffffff
# CHECK-ASM-AND-OBJ: ori a2, a3, -2048
# CHECK-ASM: encoding: [0x13,0xe6,0x06,0x80]
ori a2, a3, 0xfffff800
