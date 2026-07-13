# RUN: llvm-mc %s -triple=riscv32 -mattr=+zbkx -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zbkx -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=zbkx < %s \
# RUN:     | llvm-objdump --mattr=+zbkx -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=zbkx < %s \
# RUN:     | llvm-objdump --mattr=+zbkx -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: xperm8 t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x42,0x73,0x28]
xperm8 t0, t1, t2
# CHECK-ASM-AND-OBJ: xperm4 t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x22,0x73,0x28]
xperm4 t0, t1, t2
