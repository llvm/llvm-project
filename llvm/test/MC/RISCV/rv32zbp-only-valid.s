# With Bitmanip permutation extension:
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zbp -show-encoding \
# RUN:     | FileCheck -check-prefix=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zbp < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbp -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-OBJ %s

# CHECK-ASM: pack t0, t1, zero
# CHECK-OBJ: zext.h t0, t1
# CHECK-ASM: encoding: [0xb3,0x42,0x03,0x08]
pack t0, t1, x0
# CHECK-ASM: grevi t0, t1, 24
# CHECK-OBJ: rev8 t0, t1
# CHECK-ASM: encoding: [0x93,0x52,0x83,0x69]
grevi t0, t1, 24
