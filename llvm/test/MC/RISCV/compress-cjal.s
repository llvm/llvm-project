# RUN: llvm-mc -triple riscv32 -mattr=+c -show-encoding < %s \
# RUN:   | FileCheck -check-prefixes=CHECK,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -mattr=+c -show-encoding \
# RUN:   -M no-aliases < %s | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -triple riscv32 -mattr=+c -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+c -d - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIASOBJ %s
# RUN: llvm-mc -triple riscv32 -mattr=+c -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+c -d -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INSTOBJ %s

# c.jal is an rv32 only instruction.
jal ra, 2046
# CHECK-BYTES: 2ffd
# CHECK-ALIASOBJ: jal 0x7fe
# CHECK-ALIAS: jal 2046
# CHECK-INST: c.jal 2046
# CHECK-INSTOBJ: c.jal 0x7fe
# CHECK:  # encoding: [0xfd,0x2f]
