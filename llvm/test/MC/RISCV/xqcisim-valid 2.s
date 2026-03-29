# Xqcisim - Simulation Hint Instructions
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcisim -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcisim < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcisim -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-NOALIAS %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcisim -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST,CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcisim < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcisim --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-INST,CHECK-ALIAS %s


# CHECK-ALIAS: qc.psyscalli 1023
# CHECK-NOINST: slti zero, zero, 1023
# CHECK-ENC: encoding: [0x13,0x20,0xf0,0x3f]
qc.psyscalli 1023

# CHECK-INST: qc.pputci    255
# CHECK-ENC: encoding: [0x13,0x20,0xf0,0x4f]
qc.pputci 255

# CHECK-ALIAS: qc.c.ptrace
# CHECK-NOALIAS: c.slli zero, 0
# CHECK-ENC: encoding: [0x02,0x00]
qc.c.ptrace

# CHECK-ALIAS: qc.pcoredump
# CHECK-NOALIAS: slti zero, zero, 1536
# CHECK-ENC: encoding: [0x13,0x20,0x00,0x60]
qc.pcoredump

# CHECK-ALIAS: qc.ppregs
# CHECK-NOALIAS: slti zero, zero, 1792
# CHECK-ENC: encoding: [0x13,0x20,0x00,0x70]
qc.ppregs

# CHECK-ALIAS: qc.ppreg     a0
# CHECK-NOALIAS: slti zero, a0, -2048
# CHECK-ENC: encoding: [0x13,0x20,0x05,0x80]
qc.ppreg x10

# CHECK-ALIAS: qc.pputc     t2
# CHECK-NOALIAS: slti zero, t2, -1792
# CHECK-ENC: encoding: [0x13,0xa0,0x03,0x90]
qc.pputc x7

# CHECK-ALIAS: qc.pputs     a5
# CHECK-NOALIAS: slti zero, a5, -1536
# CHECK-ENC: encoding: [0x13,0xa0,0x07,0xa0]
qc.pputs x15

# CHECK-ALIAS: qc.pexit      s10
# CHECK-NOALIAS: slti zero, s10, -1280
# CHECK-ENC: encoding: [0x13,0x20,0x0d,0xb0]
qc.pexit x26

# CHECK-ALIAS: qc.psyscall  a1
# CHECK-NOALIAS: slti zero, a1, -1024
# CHECK-ENC: encoding: [0x13,0xa0,0x05,0xc0]
qc.psyscall x11
