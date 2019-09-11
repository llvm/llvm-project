# RUN: llvm-mc -triple riscv32 -mattr=+c -show-encoding < %s \
# RUN: | FileCheck -check-prefixes=CHECK,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -mattr=+c -show-encoding \
# RUN: -riscv-no-aliases <%s | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -triple riscv32 -mattr=+c -filetype=obj < %s \
# RUN: | llvm-objdump  -triple riscv32 -mattr=+c -d - \
# RUN: | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -mattr=+c -filetype=obj < %s \
# RUN: | llvm-objdump  -triple riscv32 -mattr=+c -d -M no-aliases - \
# RUN: | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST %s

# c.jal is an rv32 only instruction.
jal ra, 2046
# CHECK-BYTES: fd 2f
# CHECK-ALIAS: jal 2046
# CHECK-INST: c.jal 2046
# CHECK:  # encoding: [0xfd,0x2f]
