# RUN: llvm-mc -triple riscv32 -mattr=+c,+d -show-encoding < %s \
# RUN:   | FileCheck -check-prefixes=CHECK,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -mattr=+c,+d -show-encoding \
# RUN:   -riscv-no-aliases < %s | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -triple riscv32 -mattr=+c,+d -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+c,+d -d - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -mattr=+c,+d -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+c,+d -d -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST %s

# RUN: llvm-mc -triple riscv64 -mattr=+c,+d -show-encoding < %s \
# RUN:   | FileCheck -check-prefixes=CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv64 -mattr=+c,+d -show-encoding \
# RUN:   -riscv-no-aliases < %s | FileCheck -check-prefixes=CHECK-INST %s
# RUN: llvm-mc -triple riscv64 -mattr=+c,+d -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv64 --mattr=+c,+d -d - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv64 -mattr=+c,+d -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv64 --mattr=+c,+d -d -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST %s

# Tests double precision floating point instructions available in rv32 and in rv64.

fld ft0, 64(sp)
# CHECK-BYTES: 06 20
# CHECK-ALIAS: fld ft0, 64(sp)
# CHECK-INST: c.fldsp ft0, 64(sp)
# CHECK: # encoding:  [0x06,0x20]
fsd ft0, 64(sp)
# CHECK-BYTES: 82 a0
# CHECK-ALIAS: fsd ft0, 64(sp)
# CHECK-INST: c.fsdsp ft0, 64(sp)
# CHECK: # encoding:  [0x82,0xa0]
fld fs0, 248(s0)
# CHECK-BYTES: 60 3c
# CHECK-ALIAS: fld fs0, 248(s0)
# CHECK-INST: c.fld fs0, 248(s0)
# CHECK: # encoding:  [0x60,0x3c]
fsd fs0, 248(s0)
# CHECK-BYTES: 60 bc
# CHECK-ALIAS: fsd fs0, 248(s0)
# CHECK-INST: c.fsd fs0, 248(s0)
# CHECK: # encoding:  [0x60,0xbc]
