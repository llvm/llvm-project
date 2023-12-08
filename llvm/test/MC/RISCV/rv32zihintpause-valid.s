# RUN: llvm-mc %s -triple=riscv32 -mattr=+zihintpause -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zihintpause -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zihintpause < %s \
# RUN:     | llvm-objdump --mattr=+zihintpause -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zihintpause < %s \
# RUN:     | llvm-objdump --mattr=+zihintpause -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: fence w, 0
# CHECK-ASM: encoding: [0x0f,0x00,0x00,0x01]
fence w,0
# CHECK-ASM-AND-OBJ: fence 0, w
# CHECK-ASM: encoding: [0x0f,0x00,0x10,0x00]
fence 0,w
# CHECK-ASM-AND-OBJ: fence 0, 0
# CHECK-ASM: encoding: [0x0f,0x00,0x00,0x00]
fence 0,0
