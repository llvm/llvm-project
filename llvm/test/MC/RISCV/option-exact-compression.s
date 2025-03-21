# RUN: llvm-mc -triple riscv32 -show-encoding -mattr=+c %s \
# RUN:   | FileCheck -check-prefixes=CHECK,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -show-encoding -mattr=+c \
# RUN:   -M no-aliases %s | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -triple riscv32 -filetype=obj -mattr=+c %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+c --no-print-imm-hex -d - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -filetype=obj -mattr=+c %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+c --no-print-imm-hex -d -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST %s

# RUN: llvm-mc -triple riscv64 -show-encoding -mattr=+c %s \
# RUN:   | FileCheck -check-prefixes=CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv64 -show-encoding -mattr=+c \
# RUN:   -M no-aliases %s | FileCheck -check-prefixes=CHECK-INST %s
# RUN: llvm-mc -triple riscv64 -filetype=obj -mattr=+c %s \
# RUN:   | llvm-objdump  --triple=riscv64 --mattr=+c --no-print-imm-hex -d - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv64 -filetype=obj -mattr=+c %s \
# RUN:   | llvm-objdump  --triple=riscv64 --mattr=+c --no-print-imm-hex -d -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-BYTES,CHECK-INST %s


## `.option exact` disables a variety of assembler behaviour:
## - automatic compression
## - branch relaxation (of short branches to longer equivalent sequences)
## - linker relaxation (emitting R_RISCV_RELAX)
## `.option noexact` enables these behaviours again. It is also the default.

## This test only checks the automatic compression part of this behaviour.

# CHECK-BYTES: 4108
# CHECK-INST: c.lw a0, 0(a0)
# CHECK-ALIAS: lw a0, 0(a0)
# CHECK: # encoding: [0x08,0x41]
lw a0, 0(a0)

# CHECK-BYTES: 4108
# CHECK-INST: c.lw a0, 0(a0)
# CHECK-ALIAS: lw a0, 0(a0)
# CHECK: # encoding: [0x08,0x41]
c.lw a0, 0(a0)

# CHECK: .option exact
.option exact

# CHECK-BYTES: 00052503
# CHECK-INST: lw a0, 0(a0)
# CHECK-ALIAS: lw a0, 0(a0)
# CHECK: # encoding: [0x03,0x25,0x05,0x00]
lw a0, 0(a0)

# CHECK-BYTES: 4108
# CHECK-INST: c.lw a0, 0(a0)
# CHECK-ALIAS: lw a0, 0(a0)
# CHECK: # encoding: [0x08,0x41]
c.lw a0, 0(a0)

# CHECK: .option noexact
.option noexact

# CHECK-BYTES: 4108
# CHECK-INST: c.lw a0, 0(a0)
# CHECK-ALIAS: lw a0, 0(a0)
# CHECK: # encoding: [0x08,0x41]
lw a0, 0(a0)

# CHECK-BYTES: 4108
# CHECK-INST: c.lw a0, 0(a0)
# CHECK-ALIAS: lw a0, 0(a0)
# CHECK: # encoding: [0x08,0x41]
c.lw a0, 0(a0)
