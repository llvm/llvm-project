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
# RUN: not llvm-mc -triple riscv32 -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv64 -M no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s

# CHECK-ASM-AND-OBJ: lpad 22
# CHECK-ASM: auipc zero, 22
# CHECK-ASM: encoding: [0x17,0x60,0x01,0x00]
# CHECK-NO-EXT: instruction requires the following: 'Zicfilp' (Landing pad)
lpad 22
