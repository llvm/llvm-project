# With Bitmanip base extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zbb -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zbb -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zbb < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbb -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zbb < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbb -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# With Bitmanip permutation extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zbp -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zbp -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zbp < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbp -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zbp < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbp -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: andn t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x72,0x73,0x40]
andn t0, t1, t2
# CHECK-ASM-AND-OBJ: orn t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x62,0x73,0x40]
orn t0, t1, t2
# CHECK-ASM-AND-OBJ: xnor t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x42,0x73,0x40]
xnor t0, t1, t2
# CHECK-ASM-AND-OBJ: rol t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x12,0x73,0x60]
rol t0, t1, t2
# CHECK-ASM-AND-OBJ: ror t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x52,0x73,0x60]
ror t0, t1, t2
# CHECK-ASM-AND-OBJ: rori t0, t1, 31
# CHECK-ASM: encoding: [0x93,0x52,0xf3,0x61]
rori t0, t1, 31
# CHECK-ASM-AND-OBJ: rori t0, t1, 0
# CHECK-ASM: encoding: [0x93,0x52,0x03,0x60]
rori t0, t1, 0
# CHECK-ASM-AND-OBJ: orc.b t0, t1
# CHECK-ASM: encoding: [0x93,0x52,0x73,0x28]
orc.b t0, t1
