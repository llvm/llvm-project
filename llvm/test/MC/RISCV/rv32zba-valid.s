# With Bitmanip base extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zba -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zba -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zba < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zba -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zba < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zba -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: sh1add t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x22,0x73,0x20]
sh1add t0, t1, t2
# CHECK-ASM-AND-OBJ: sh2add t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x42,0x73,0x20]
sh2add t0, t1, t2
# CHECK-ASM-AND-OBJ: sh3add t0, t1, t2
# CHECK-ASM: encoding: [0xb3,0x62,0x73,0x20]
sh3add t0, t1, t2
