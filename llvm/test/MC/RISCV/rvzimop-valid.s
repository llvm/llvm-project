# RUN: llvm-mc %s -triple=riscv32 -mattr=+zimop -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zimop -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zimop < %s \
# RUN:     | llvm-objdump --mattr=+zimop -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zimop < %s \
# RUN:     | llvm-objdump --mattr=+zimop -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: mop.r.0 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xc5,0x81]
mop.r.0 a2, a1

# CHECK-ASM-AND-OBJ: mop.r.31 a2, a1
# CHECK-ASM: encoding: [0x73,0xc6,0xf5,0xcd]
mop.r.31 a2, a1

# CHECK-ASM-AND-OBJ: mop.rr.0 a3, a2, a1
# CHECK-ASM: encoding: [0xf3,0x46,0xb6,0x82]
mop.rr.0 a3, a2, a1

# CHECK-ASM-AND-OBJ: mop.rr.7 a3, a2, a1
# CHECK-ASM: encoding: [0xf3,0x46,0xb6,0xce]
mop.rr.7 a3, a2, a1