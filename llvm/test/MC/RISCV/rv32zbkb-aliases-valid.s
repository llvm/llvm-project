# RUN: llvm-mc %s -triple=riscv32 -mattr=+zbkb -M no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+zbkb \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zbkb < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r -M no-aliases --mattr=+zbkb - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+zbkb < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r --mattr=+zbkb - \
# RUN:     | FileCheck -check-prefixes=CHECK-S-OBJ %s

# The following check prefixes are used in this test:
# CHECK-S-OBJ            Match both the .s and objdumped object output with
#                        aliases enabled
# CHECK-S-OBJ-NOALIAS    Match both the .s and objdumped object output with
#                        aliases disabled

# CHECK-S-OBJ-NOALIAS: rori t0, t1, 8
# CHECK-S-OBJ: rori t0, t1, 8
ror x5, x6, 8
