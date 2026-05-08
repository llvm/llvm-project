# RUN: llvm-mc %s -triple=riscv32 -mattr=+zicbop -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zicbop -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zicbop < %s \
# RUN:     | llvm-objdump --no-print-imm-hex --mattr=+zicbop -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zicbop < %s \
# RUN:     | llvm-objdump --no-print-imm-hex --mattr=+zicbop -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s

# prefetch hints are always available even without Zicbop
# (riscv-non-isa/riscv-elf-psabi-doc#474).

# RUN: llvm-mc %s -triple=riscv32 -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: prefetch.i -2048(t0)
# CHECK-ASM: encoding: [0x13,0xe0,0x02,0x80]
prefetch.i -2048(t0)
# CHECK-ASM-AND-OBJ: prefetch.i 2016(t0)
# CHECK-ASM: encoding: [0x13,0xe0,0x02,0x7e]
prefetch.i 2016(t0)

# CHECK-ASM-AND-OBJ: prefetch.r -2048(t1)
# CHECK-ASM: encoding: [0x13,0x60,0x13,0x80]
prefetch.r -0x800(t1)
# CHECK-ASM-AND-OBJ: prefetch.r 2016(t1)
# CHECK-ASM: encoding: [0x13,0x60,0x13,0x7e]
prefetch.r 0x7e0(t1)

# CHECK-ASM-AND-OBJ: prefetch.w -2048(t2)
# CHECK-ASM: encoding: [0x13,0xe0,0x33,0x80]
prefetch.w -2048(t2)
# CHECK-ASM-AND-OBJ: prefetch.w 2016(t2)
# CHECK-ASM: encoding: [0x13,0xe0,0x33,0x7e]
prefetch.w 2016(t2)

# Ensure that ori disassembles as prefetch.*
# FIXME: Should CHECK-ASM also print prefetch.*?

# CHECK-ASM: ori zero, t0, -2048
# CHECK-OBJ: prefetch.i -2048(t0)
# CHECK-ASM: encoding: [0x13,0xe0,0x02,0x80]
ori x0, t0, -2048
# CHECK-ASM: ori zero, t0, 2016
# CHECK-OBJ: prefetch.i 2016(t0)
# CHECK-ASM: encoding: [0x13,0xe0,0x02,0x7e]
ori x0, t0, 2016

# CHECK-ASM: ori zero, t1, -2047
# CHECK-OBJ: prefetch.r -2048(t1)
# CHECK-ASM: encoding: [0x13,0x60,0x13,0x80]
ori x0, t1, -2047
# CHECK-ASM: ori zero, t1, 2017
# CHECK-OBJ: prefetch.r 2016(t1)
# CHECK-ASM: encoding: [0x13,0x60,0x13,0x7e]
ori x0, t1, 2017

# CHECK-ASM: ori zero, t2, -2045
# CHECK-OBJ: prefetch.w -2048(t2)
# CHECK-ASM: encoding: [0x13,0xe0,0x33,0x80]
ori x0, t2, -2045
# CHECK-ASM: ori zero, t2, 2019
# CHECK-OBJ: prefetch.w 2016(t2)
# CHECK-ASM: encoding: [0x13,0xe0,0x33,0x7e]
ori x0, t2, 2019

# Ensure that enabling zicbop doesn't cause issues decoding ori instructions.

# CHECK-ASM-AND-OBJ: ori a0, a1, -2048
# CHECK-ASM: encoding: [0x13,0xe5,0x05,0x80]
ori a0, a1, -2048
# CHECK-ASM-AND-OBJ: ori a0, a1, 0
# CHECK-ASM: encoding: [0x13,0xe5,0x05,0x00]
ori a0, a1, 0
