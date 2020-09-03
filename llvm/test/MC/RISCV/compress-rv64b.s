# RUN: llvm-mc -triple riscv64 -mattr=+c,+experimental-zbproposedc,+experimental-b -show-encoding < %s \
# RUN: | FileCheck -check-prefixes=CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv64 -mattr=+c,+experimental-zbproposedc,+experimental-b -show-encoding \
# RUN: -riscv-no-aliases <%s | FileCheck -check-prefixes=CHECK-INST %s
# RUN: llvm-mc -triple riscv64 -mattr=+c,+experimental-zbproposedc,+experimental-b -filetype=obj < %s \
# RUN: | llvm-objdump  --triple=riscv64 --mattr=+c,+experimental-zbproposedc,+experimental-b -d - \
# RUN: | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv64 -mattr=+c,+experimental-zbproposedc,+experimental-b -filetype=obj < %s \
# RUN: | llvm-objdump  --triple=riscv64 --mattr=+c,+experimental-zbproposedc,+experimental-b -d -M no-aliases - \
# RUN: | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST %s

# Tests compressed instructions available in rv64 and not in rv32.

# CHECK-BYTES: 01 68
# CHECK-ALIAS: zext.w s0, s0
# CHECK-INST: c.zext.w s0
# CHECK: # encoding:  [0x01,0x68]
zext.w s0, s0
