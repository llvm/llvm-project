# With Bitmanip base extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+zbb -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zbb < %s \
# RUN:     | llvm-objdump --mattr=+zbb -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: clz t0, t1
# CHECK-ASM: encoding: [0x93,0x12,0x03,0x60]
clz t0, t1
# CHECK-ASM-AND-OBJ: ctz t0, t1
# CHECK-ASM: encoding: [0x93,0x12,0x13,0x60]
ctz t0, t1
# CHECK-ASM-AND-OBJ: cpop t0, t1
# CHECK-ASM: encoding: [0x93,0x12,0x23,0x60]
cpop t0, t1

# CHECK-ASM-AND-OBJ: sext.b t0, t1
# CHECK-ASM: encoding: [0x93,0x12,0x43,0x60]
sext.b t0, t1
# CHECK-ASM-AND-OBJ: sext.h t0, t1
# CHECK-ASM: encoding: [0x93,0x12,0x53,0x60]
sext.h t0, t1
# CHECK-ASM-AND-OBJ: zext.h t0, t1
# CHECK-ASM: encoding: [0xb3,0x42,0x03,0x08]
zext.h t0, t1

# CHECK-ASM-AND-OBJ: min t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x42,0x73,0x0a]
min t0, t1, t2
# CHECK-ASM-AND-OBJ: minu t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x52,0x73,0x0a]
minu t0, t1, t2
# CHECK-ASM-AND-OBJ: max t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x62,0x73,0x0a]
max t0, t1, t2
# CHECK-ASM-AND-OBJ: maxu t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x72,0x73,0x0a]
maxu t0, t1, t2
