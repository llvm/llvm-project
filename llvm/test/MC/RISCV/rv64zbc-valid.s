# With B extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-b -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-b < %s \
# RUN:     | llvm-objdump --mattr=+experimental-b -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# With Bitmanip carry-less multiply extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zbc -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zbc < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbc -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: clmulw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x12,0x73,0x0a]
clmulw t0, t1, t2
# CHECK-ASM-AND-OBJ: clmulrw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x22,0x73,0x0a]
clmulrw t0, t1, t2
# CHECK-ASM-AND-OBJ: clmulhw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x32,0x73,0x0a]
clmulhw t0, t1, t2
