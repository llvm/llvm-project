# RUN: llvm-mc %s -triple=riscv32 -mattr=+zbs -M no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+zbs \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zbs < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r -M no-aliases --mattr=+zbs - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zbs < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r --mattr=+zbs - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s

# The following check prefixes are used in this test:
# CHECK-S-OBJ            Match both the .s and objdumped object output with
#                        aliases enabled
# CHECK-S-OBJ-NOALIAS    Match both the .s and objdumped object output with
#                        aliases disabled

# CHECK-S-OBJ-NOALIAS: bseti t0, t1, 8
# CHECK-S-OBJ: bseti t0, t1, 8
bset x5, x6, 8

# CHECK-S-OBJ-NOALIAS: bclri t0, t1, 8
# CHECK-S-OBJ: bclri t0, t1, 8
bclr x5, x6, 8

# CHECK-S-OBJ-NOALIAS: binvi t0, t1, 8
# CHECK-S-OBJ: binvi t0, t1, 8
binv x5, x6, 8

# CHECK-S-OBJ-NOALIAS: bexti t0, t1, 8
# CHECK-S-OBJ: bexti t0, t1, 8
bext x5, x6, 8

# CHECK-S-OBJ-NOALIAS: bseti t2, zero, 11
# CHECK-S-OBJ: bseti t2, zero, 11
li x7, 2048
