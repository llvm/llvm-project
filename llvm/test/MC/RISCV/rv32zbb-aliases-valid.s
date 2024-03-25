# RUN: llvm-mc %s -triple=riscv32 -mattr=+zbb -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+zbb \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zbb < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r -M no-aliases --mattr=+zbb - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zbb < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r --mattr=+zbb - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s

# The following check prefixes are used in this test:
# CHECK-S-OBJ            Match both the .s and objdumped object output with
#                        aliases enabled
# CHECK-S-OBJ-NOALIAS    Match both the .s and objdumped object output with
#                        aliases disabled

# CHECK-S-OBJ-NOALIAS: zext.h t0, t1
# CHECK-S-OBJ: zext.h t0, t1
zext.h x5, x6

# CHECK-S-OBJ-NOALIAS: rev8 t0, t1
# CHECK-S-OBJ: rev8 t0, t1
rev8 x5, x6

# CHECK-S-OBJ-NOALIAS: orc.b t0, t1
# CHECK-S-OBJ: orc.b t0, t1
orc.b x5, x6

# CHECK-S-OBJ-NOALIAS: rori t0, t1, 8
# CHECK-S-OBJ: rori t0, t1, 8
ror x5, x6, 8
