# RUN: llvm-mc -triple riscv32 -mattr=+c,+f -show-encoding < %s \
# RUN:   | FileCheck -check-prefixes=CHECK,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -mattr=+c,+f -show-encoding \
# RUN:   -riscv-no-aliases < %s | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -triple riscv32 -mattr=+c,+f -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+c,+f --no-print-imm-hex -d - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -mattr=+c,+f -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+c,+f --no-print-imm-hex -d -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST %s
# RUN: llvm-mc -triple riscv32 -mattr=+zcf,+f -show-encoding < %s \
# RUN:   | FileCheck -check-prefixes=CHECK,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -mattr=+zcf,+f -show-encoding \
# RUN:   -riscv-no-aliases < %s | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -triple riscv32 -mattr=+zcf,+f -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+zcf,+f --no-print-imm-hex -d - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -mattr=+zcf,+f -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+zcf,+f --no-print-imm-hex -d -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST %s

# Instructions that are 32 bit only.
flw ft0, 124(sp)
# CHECK-BYTES: 76 70
# CHECK-ALIAS: flw     ft0, 124(sp)
# CHECK-INST: c.flwsp ft0, 124(sp)
# CHECK:  # encoding: [0x76,0x70]
fsw ft0, 124(sp)
# CHECK-BYTES: 82 fe
# CHECK-ALIAS: fsw ft0, 124(sp)
# CHECK-INST: c.fswsp ft0, 124(sp)
# CHECK:  # encoding: [0x82,0xfe]
flw fs0, 124(s0)
# CHECK-BYTES: 60 7c
# CHECK-ALIAS: flw fs0, 124(s0)
# CHECK-INST: c.flw fs0, 124(s0)
# CHECK:  # encoding:  [0x60,0x7c]
fsw fs0, 124(s0)
# CHECK-BYTES: 60 fc
# CHECK-ALIAS: fsw fs0, 124(s0)
# CHECK-INST: c.fsw fs0, 124(s0)
# CHECK:  # encoding:  [0x60,0xfc]
