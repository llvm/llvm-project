# RUN: llvm-mc %s -triple=riscv32 -mattr=+zbkb -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zbkb -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zbkb < %s \
# RUN:     | llvm-objdump --mattr=+zbkb -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zbkb < %s \
# RUN:     | llvm-objdump --mattr=+zbkb -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: ror t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x52,0x73,0x60]
ror t0, t1, t2
# CHECK-ASM-AND-OBJ: rol t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x12,0x73,0x60]
rol t0, t1, t2
# CHECK-ASM-AND-OBJ: rori t0, t1, 31
# CHECK-ASM: encoding: [0x93,0x52,0xf3,0x61]
rori t0, t1, 31
# CHECK-ASM-AND-OBJ: rori t0, t1, 0
# CHECK-ASM: encoding: [0x93,0x52,0x03,0x60]
rori t0, t1, 0

# CHECK-ASM-AND-OBJ: andn t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x72,0x73,0x40]
andn t0, t1, t2
# CHECK-ASM-AND-OBJ: orn t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x62,0x73,0x40]
orn t0, t1, t2
# CHECK-ASM-AND-OBJ: xnor t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x42,0x73,0x40]
xnor t0, t1, t2

# CHECK-ASM-AND-OBJ: pack t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x42,0x73,0x08]
pack t0, t1, t2

# Test the encoding used for zext.h for RV32.
# CHECK-ASM-AND-OBJ: pack t0, t1, zero
# CHECK-ASM: encoding: [0xb3,0x42,0x03,0x08]
pack t0, t1, x0

# CHECK-ASM-AND-OBJ: packh t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x72,0x73,0x08]
packh t0, t1, t2

# CHECK-ASM-AND-OBJ: brev8 t0, t1
# CHECK-ASM: encoding: [0x93,0x52,0x73,0x68]
brev8 t0, t1
