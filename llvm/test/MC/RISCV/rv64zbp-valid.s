# With B extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-b -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-b < %s \
# RUN:     | llvm-objdump --mattr=+experimental-b -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# With Bitmanip permutation extension:
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zbp -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zbp < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zbp -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: gorcw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x52,0x73,0x28]
gorcw t0, t1, t2
# CHECK-ASM-AND-OBJ: grevw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x52,0x73,0x68]
grevw t0, t1, t2
# CHECK-ASM-AND-OBJ: gorciw t0, t1, 0
# CHECK-ASM: encoding: [0x9b,0x52,0x03,0x28]
gorciw t0, t1, 0
# CHECK-ASM-AND-OBJ: greviw t0, t1, 0
# CHECK-ASM: encoding: [0x9b,0x52,0x03,0x68]
greviw t0, t1, 0
# CHECK-ASM-AND-OBJ: shflw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x12,0x73,0x08]
shflw t0, t1, t2
# CHECK-ASM-AND-OBJ: unshflw t0, t1, t2
# CHECK-ASM: encoding: [0xbb,0x52,0x73,0x08]
unshflw t0, t1, t2
