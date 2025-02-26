# Xqcisim - Simulation Hint Instructions
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcisim -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcisim < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcisim -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcisim -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcisim < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcisim --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s


# CHECK-INST: qc.psyscalli 1023
# CHECK-ENC: encoding: [0x13,0x20,0xf0,0x3f]
qc.psyscalli 1023

# CHECK-INST: qc.pputci    255
# CHECK-ENC: encoding: [0x13,0x20,0xf0,0x4f]
qc.pputci 255

# CHECK-INST: qc.c.ptrace
# CHECK-ENC: encoding: [0x02,0x00]
qc.c.ptrace

# CHECK-INST: qc.pcoredump
# CHECK-ENC: encoding: [0x13,0x20,0x00,0x60]
qc.pcoredump

# CHECK-INST: qc.ppregs
# CHECK-ENC: encoding: [0x13,0x20,0x00,0x70]
qc.ppregs

# CHECK-INST: qc.ppreg     a0
# CHECK-ENC: encoding: [0x13,0x20,0x05,0x80]
qc.ppreg x10

# CHECK-INST: qc.pputc     t2
# CHECK-ENC: encoding: [0x13,0xa0,0x03,0x90]
qc.pputc x7

# CHECK-INST: qc.pputs     a5
# CHECK-ENC: encoding: [0x13,0xa0,0x07,0xa0]
qc.pputs x15

# CHECK-INST: qc.pexit      s10
# CHECK-ENC: encoding: [0x13,0x20,0x0d,0xb0]
qc.pexit x26

# CHECK-INST: qc.psyscall  a1
# CHECK-ENC: encoding: [0x13,0xa0,0x05,0xc0]
qc.psyscall x11
