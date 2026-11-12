# RUN: llvm-mc %s -triple=riscv32 -mattr=+zicboz -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zicboz -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zicboz < %s \
# RUN:     | llvm-objdump --mattr=+zicboz -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zicboz < %s \
# RUN:     | llvm-objdump --mattr=+zicboz -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: cbo.zero (t0)
# CHECK-ASM: encoding: [0x0f,0xa0,0x42,0x00]
cbo.zero (t0)
# CHECK-ASM-AND-OBJ: cbo.zero (t0)
# CHECK-ASM: encoding: [0x0f,0xa0,0x42,0x00]
cbo.zero 0(t0)
