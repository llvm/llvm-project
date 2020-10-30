# RUN: llvm-mc %s -triple=riscv64 -mattr=+c,+experimental-zbproposedc,+experimental-b -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+experimental-zbproposedc,+experimental-b < %s \
# RUN:     | llvm-objdump --mattr=+c,+experimental-zbproposedc,+experimental-b -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s


# CHECK-ASM-AND-OBJ: c.zext.w s0
# CHECK-ASM: encoding: [0x01,0x68]
c.zext.w s0
