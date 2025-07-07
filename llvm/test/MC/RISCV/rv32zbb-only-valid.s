# With Bitmanip base extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+zbb -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zbb < %s \
# RUN:     | llvm-objdump --mattr=+zbb --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: zext.h t0, t1
# CHECK-ASM: encoding: [0xb3,0x42,0x03,0x08]
zext.h t0, t1
# CHECK-ASM-AND-OBJ: rev8 t0, t1
# CHECK-ASM: encoding: [0x93,0x52,0x83,0x69]
rev8 t0, t1
