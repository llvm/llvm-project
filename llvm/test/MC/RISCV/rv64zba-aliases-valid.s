# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zba -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s  -triple=riscv64 -mattr=+experimental-zba \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+experimental-zba < %s \
# RUN:     | llvm-objdump -d -r -M no-aliases --mattr=+experimental-zba - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+experimental-zba < %s \
# RUN:     | llvm-objdump -d -r --mattr=+experimental-zba - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s

# The following check prefixes are used in this test:
# CHECK-S-OBJ            Match both the .s and objdumped object output with
#                        aliases enabled
# CHECK-S-OBJ-NOALIAS    Match both the .s and objdumped object output with
#                        aliases disabled

# CHECK-S-OBJ-NOALIAS: add.uw t0, t1, zero
# CHECK-S-OBJ: zext.w t0, t1
zext.w x5, x6

# CHECK-S-OBJ-NOALIAS: addi t1, zero, -2
# CHECK-S-OBJ-NOALIAS-NEXT: add.uw t1, t1, zero
# CHECK-S-OBJ: addi t1, zero, -2
# CHECK-S-OBJ-NEXT: zext.w t1, t1
li x6, 0xfffffffe

# CHECK-S-OBJ-NOALIAS: lui t2, 699051
# CHECK-S-OBJ-NOALIAS-NEXT: addiw t2, t2, -1366
# CHECK-S-OBJ-NOALIAS-NEXT: add.uw t2, t2, zero
# CHECK-S-OBJ: lui t2, 699051
# CHECK-S-OBJ-NEXT: addiw t2, t2, -1366
# CHECK-S-OBJ-NEXT: zext.w t2, t2
li x7, 0xaaaaaaaa
