# RUN: llvm-mc %s -triple=riscv32 -mattr=+c -M no-aliases -show-encoding \
# RUN:     | FileCheck --check-prefix=CHECK-ASM %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c < %s \
# RUN:     | llvm-objdump --mattr=+c --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-OBJ %s
#
# RUN: llvm-mc --verify=no-ext -triple riscv32 -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify=no-rv32 -triple riscv64 -mattr=+c -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify=no-rv32-and-ext -triple riscv64 -M no-aliases -show-encoding < %s

# CHECK-OBJ: c.jal 0x7fe
# CHECK-ASM: c.jal 2046
# CHECK-ASM: encoding: [0xfd,0x2f]
c.jal 2046
# no-ext-error@-1 {{instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores)}}
# no-rv32-error@-2 {{instruction requires the following: RV32I Base Instruction Set}}
# no-rv32-and-ext-error@-3 {{instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores), RV32I Base Instruction Set}}

# CHECK-OBJ: c.addi a1, -1
# CHECK-ASM: c.addi a1, -1
# CHECK-ASM: encoding: [0xfd,0x15]
c.addi a1, 0xffffffff
# no-ext-error@-1 {{instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores)}}
# no-rv32-error@-2 {{immediate must be an integer in the range [-32, 31]}}
# no-rv32-and-ext-error@-3 {{instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores)}}

# CHECK-OBJ: c.addi16sp sp, -352
# CHECK-ASM: c.addi16sp sp, -352
# CHECK-ASM: encoding: [0x0d,0x71]
c.addi16sp sp, 0xfffffea0
# no-ext-error@-1 {{instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores)}}
# no-rv32-error@-2 {{immediate must be a multiple of 16 bytes and non-zero in the range [-512, 496]}}
# no-rv32-and-ext-error@-3 {{invalid instruction}}

## Branch and Jump immediates are relative but printed as their absolute address
## when disassembling.

# CHECK-OBJ: c.beqz a2, 0xffffff06
# CHECK-ASM: c.beqz a2, -256
# CHECK-ASM: encoding: [0x01,0xd2]
c.beqz a2, 0xffffff00
# no-ext-error@-1 {{instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores)}}
# no-rv32-error@-2 {{immediate must be a multiple of 2 bytes in the range [-256, 254]}}
# no-rv32-and-ext-error@-3 {{invalid instruction}}

# CHECK-OBJ: c.beqz a0, 0xffffff16
# CHECK-ASM: .insn cb 1, 6, a0, -242
# CHECK-ASM: encoding: [0x19,0xd5]
.insn cb 1, 6, a0, 0xffffff0e
# no-ext-error@-1 {{invalid instruction format}}
# no-rv32-error@-2 {{immediate must be a multiple of 2 bytes in the range [-256, 254]}}
# no-rv32-and-ext-error@-3 {{invalid instruction format}}

# CHECK-OBJ: c.jal 0xfffffab4
# CHECK-ASM: c.jal -1366
# CHECK-ASM: encoding: [0x6d,0x34]
c.jal 0xfffffaaa
# no-ext-error@-1 {{instruction requires the following: 'C' (Compressed Instructions) or 'Zca' (part of the C extension, excluding compressed floating point loads/stores)}}
# no-rv32-error@-2 {{invalid instruction}}
# no-rv32-and-ext-error@-3 {{invalid instruction}}

# CHECK-OBJ: c.j 0xfffffcd8
# CHECK-ASM: .insn cj 1, 5, -820
# CHECK-ASM: encoding: [0xf1,0xb1]
.insn cj 1, 5, 0xfffffccc
# no-ext-error@-1 {{invalid instruction format}}
# no-rv32-error@-2 {{immediate must be a multiple of 2 bytes in the range [-2048, 2046]}}
# no-rv32-and-ext-error@-3 {{invalid instruction format}}
