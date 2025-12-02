# RUN: llvm-mc %s -triple=riscv32 -mattr=+zicond -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zicond -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zicond < %s \
# RUN:     | llvm-objdump --mattr=+zicond -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zicond < %s \
# RUN:     | llvm-objdump --mattr=+zicond -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: czero.eqz t0, a3, ra
# CHECK-ASM: encoding: [0xb3,0xd2,0x16,0x0e]
czero.eqz t0, a3, ra

# CHECK-ASM-AND-OBJ: czero.nez a1, gp, t6
# CHECK-ASM: encoding: [0xb3,0xf5,0xf1,0x0f]
czero.nez a1, gp, t6
