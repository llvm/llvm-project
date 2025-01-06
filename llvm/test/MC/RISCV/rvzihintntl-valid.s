# RUN: llvm-mc %s -triple=riscv32 -mattr=+zihintntl -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zihintntl -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zihintntl < %s \
# RUN:     | llvm-objdump --mattr=+zihintntl -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zihintntl < %s \
# RUN:     | llvm-objdump --mattr=+zihintntl -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: add zero, zero, sp
# CHECK-ASM: encoding: [0x33,0x00,0x20,0x00]
ntl.p1

# CHECK-ASM-AND-OBJ: add zero, zero, gp
# CHECK-ASM: encoding: [0x33,0x00,0x30,0x00]
ntl.pall

# CHECK-ASM-AND-OBJ: add zero, zero, tp
# CHECK-ASM: encoding: [0x33,0x00,0x40,0x00]
ntl.s1

# CHECK-ASM-AND-OBJ: add zero, zero, t0
# CHECK-ASM: encoding: [0x33,0x00,0x50,0x00]
ntl.all
