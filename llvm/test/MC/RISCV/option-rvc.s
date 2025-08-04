# RUN: llvm-mc -triple riscv32 -show-encoding < %s \
# RUN:   | FileCheck -check-prefixes=CHECK,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -show-encoding \
# RUN:   -M no-aliases < %s | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -triple riscv32 -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+c --no-print-imm-hex -d - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+c --no-print-imm-hex -d -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST %s

# RUN: llvm-mc -triple riscv64 -show-encoding < %s \
# RUN:   | FileCheck -check-prefixes=CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv64 -show-encoding \
# RUN:   -M no-aliases < %s | FileCheck -check-prefixes=CHECK-INST %s
# RUN: llvm-mc -triple riscv64 -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv64 --mattr=+c --no-print-imm-hex -d - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv64 -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv64 --mattr=+c --no-print-imm-hex -d -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST %s

# CHECK-BYTES: 00058513
# CHECK-ALIAS: mv a0, a1
# CHECK-INST: addi a0, a1, 0
# CHECK: # encoding:  [0x13,0x85,0x05,0x00]
addi a0, a1, 0

# CHECK-BYTES: 3fc10413
# CHECK-ALIAS: addi s0, sp, 1020
# CHECK-INST: addi s0, sp, 1020
# CHECK: # encoding:  [0x13,0x04,0xc1,0x3f]
addi s0, sp, 1020


# CHECK: .option rvc
.option rvc
# CHECK-BYTES: 852e
# CHECK-ALIAS: mv a0, a1
# CHECK-INST: c.mv a0, a1
# CHECK: # encoding:  [0x2e,0x85]
addi a0, a1, 0

# CHECK-BYTES: 1fe0
# CHECK-ALIAS: addi s0, sp, 1020
# CHECK-INST: c.addi4spn s0, sp, 1020
# CHECK: # encoding:  [0xe0,0x1f]
addi s0, sp, 1020

# CHECK: .option norvc
.option norvc
# CHECK-BYTES: 00058513
# CHECK-ALIAS: mv a0, a1
# CHECK-INST: addi a0, a1, 0
# CHECK: # encoding:  [0x13,0x85,0x05,0x00]
addi a0, a1, 0

# CHECK-BYTES: 3fc10413
# CHECK-ALIAS: addi s0, sp, 1020
# CHECK-INST: addi s0, sp, 1020
# CHECK: # encoding:  [0x13,0x04,0xc1,0x3f]
addi s0, sp, 1020

# CHECK: .option rvc
.option rvc
# CHECK-BYTES: 852e
# CHECK-ALIAS: mv a0, a1
# CHECK-INST: c.mv a0, a1
# CHECK: # encoding:  [0x2e,0x85]
addi a0, a1, 0

# CHECK-BYTES: 1fe0
# CHECK-ALIAS: addi s0, sp, 1020
# CHECK-INST: c.addi4spn s0, sp, 1020
# CHECK: # encoding:  [0xe0,0x1f]
addi s0, sp, 1020

# CHECK: .option norvc
.option norvc
# CHECK-BYTES: 00058513
# CHECK-ALIAS: mv a0, a1
# CHECK-INST: addi a0, a1, 0
# CHECK: # encoding:  [0x13,0x85,0x05,0x00]
addi a0, a1, 0

# CHECK-BYTES: 3fc10413
# CHECK-ALIAS: addi s0, sp, 1020
# CHECK-INST: addi s0, sp, 1020
# CHECK: # encoding:  [0x13,0x04,0xc1,0x3f]
addi s0, sp, 1020
