# RUN: llvm-mc %s -triple=riscv32 -mattr=+c -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck --check-prefix=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c < %s \
# RUN:     | llvm-objdump --mattr=+c -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck --check-prefix=CHECK-NO-EXT %s
# RUN: not llvm-mc -triple riscv64 -mattr=+c \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck --check-prefix=CHECK-NO-RV32 %s
# RUN: not llvm-mc -triple riscv64 \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck --check-prefix=CHECK-NO-RV32-AND-EXT %s

# CHECK-OBJ: c.jal 0x7fe
# CHECK-ASM: c.jal 2046
# CHECK-ASM: encoding: [0xfd,0x2f]
# CHECK-NO-EXT: error: instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores){{$}}
# CHECK-NO-RV32: error: instruction requires the following: RV32I Base Instruction Set{{$}}
# CHECK-NO-RV32-AND-EXT: error: instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores), RV32I Base Instruction Set{{$}}
c.jal 2046

# CHECK-OBJ: c.addi a1, -1
# CHECK-ASM: c.addi a1, -1
# CHECK-ASM: encoding: [0xfd,0x15]
c.addi a1, 0xffffffff
