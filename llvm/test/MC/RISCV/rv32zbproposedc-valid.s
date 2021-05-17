# RUN: llvm-mc %s -triple=riscv32 -mattr=+c,+experimental-zbproposedc -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+c,+experimental-zbproposedc,+experimental-b -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c,+experimental-zbproposedc < %s \
# RUN:     | llvm-objdump --mattr=+c,+experimental-zbproposedc -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+experimental-zbproposedc,+experimental-b < %s \
# RUN:     | llvm-objdump --mattr=+c,+experimental-zbproposedc,+experimental-b -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: c.not s0
# CHECK-ASM: encoding: [0x01,0x60]
c.not s0
# CHECK-ASM-AND-OBJ: c.neg s0
# CHECK-ASM: encoding: [0x01,0x64]
c.neg s0
