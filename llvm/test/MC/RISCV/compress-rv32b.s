# RUN: llvm-mc -triple riscv32 -mattr=+c,+experimental-zbproposedc -show-encoding < %s \
# RUN:   | FileCheck -check-prefixes=CHECK,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -mattr=+c,+experimental-zbproposedc -show-encoding \
# RUN:   -riscv-no-aliases < %s | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -triple riscv32 -mattr=+c,+experimental-zbproposedc -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+c,+experimental-zbproposedc -d - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -mattr=+c,+experimental-zbproposedc -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+c,+experimental-zbproposedc -d -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST %s

# RUN: llvm-mc -triple riscv64 -mattr=+c,+experimental-zbproposedc -show-encoding < %s \
# RUN:   | FileCheck -check-prefixes=CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv64 -mattr=+c,+experimental-zbproposedc -show-encoding \
# RUN:   -riscv-no-aliases < %s | FileCheck -check-prefixes=CHECK-INST %s
# RUN: llvm-mc -triple riscv64 -mattr=+c,+experimental-zbproposedc -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv64 --mattr=+c,+experimental-zbproposedc -d - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv64 -mattr=+c,+experimental-zbproposedc -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv64 --mattr=+c,+experimental-zbproposedc -d -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST %s

# Tests bit manipulation instructions available in rv32 and in rv64.

# CHECK-BYTES: 01 60
# CHECK-ALIAS: not s0, s0
# CHECK-INST: c.not s0
# CHECK: # encoding:  [0x01,0x60]
not s0, s0

# CHECK-BYTES: 01 64
# CHECK-ALIAS: neg s0, s0
# CHECK-INST: c.neg s0
# CHECK: # encoding:  [0x01,0x64]
neg s0, s0
