# RUN: llvm-mc %s -triple=riscv32 -mattr=+zbkb -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zbkb < %s \
# RUN:     | llvm-objdump --mattr=+zbkb -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: rev8 t0, t1
# CHECK-ASM: encoding: [0x93,0x52,0x83,0x69]
rev8 t0, t1

# CHECK-ASM-AND-OBJ: zip t0, t1
# CHECK-ASM: encoding: [0x93,0x12,0xf3,0x08]
zip t0, t1
# CHECK-S-OBJ-NOALIAS: unzip t0, t1
# CHECK-ASM: encoding: [0x93,0x52,0xf3,0x08]
unzip t0, t1

# Test the encoding used for zext.h for RV32.
# CHECK-ASM-AND-OBJ: zext.h t0, t1
# CHECK-ASM: encoding: [0xb3,0x42,0x03,0x08]
pack t0, t1, x0

# CHECK-ASM-AND-OBJ: zext.h t0, t1
# CHECK-ASM: encoding: [0xb3,0x42,0x03,0x08]
zext.h t0, t1
