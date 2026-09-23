# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zicfilp -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zicfilp -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zicfilp < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zicfilp --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zicfilp < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zicfilp --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# lpad is always available in the assembler and disassembler, even without
# enabling Zicfilp (riscv-non-isa/riscv-elf-psabi-doc#474).
#
# RUN: llvm-mc %s -triple=riscv32 -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc %s -triple=riscv64 -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 < %s \
# RUN:     | llvm-objdump --no-print-imm-hex -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: lpad 22
# CHECK-ASM: auipc zero, 22
# CHECK-ASM: encoding: [0x17,0x60,0x01,0x00]
lpad 22
