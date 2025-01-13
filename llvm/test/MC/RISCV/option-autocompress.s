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


# `.option (no)autocompress` enables and disables instruction compression in the
# assembler, without changing the current architecture.
#
# The default is as if `.option autocompress` has been specified, that is, the
# assembler compresses by default.

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

# CHECK: .option noautocompress
.option noautocompress

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

# CHECK: .option autocompress
.option autocompress

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
