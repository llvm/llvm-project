# RUN: llvm-mc %s -triple=riscv32 -mattr=+c,+f -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c,+f < %s \
# RUN:     | llvm-objdump --mattr=+c,+f --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+zcf,+f -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+zcf,+f < %s \
# RUN:     | llvm-objdump --mattr=+zcf,+f --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify=no-ext -triple riscv32 -mattr=+c -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify=no-ext -triple riscv32 -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify=no-rv32 -triple riscv64 -mattr=+c,+f -M no-aliases -show-encoding < %s

# FIXME: error messages for rv64fc are misleading

# CHECK-ASM-AND-OBJ: c.flwsp  fs0, 252(sp)
# CHECK-ASM: encoding: [0x7e,0x74]
c.flwsp  fs0, 252(sp)
# no-ext-error@-1 {{instruction requires the following: 'C' (Compressed Instructions) and 'F' (Single-Precision Floating-Point) or 'Zcf' (Compressed Single-Precision Floating-Point Instructions)}}
# no-rv32-error@-2 {{instruction requires the following: 'C' (Compressed Instructions) and 'F' (Single-Precision Floating-Point) or 'Zcf' (Compressed Single-Precision Floating-Point Instructions), RV32I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: c.fswsp  fa7, 252(sp)
# CHECK-ASM: encoding: [0xc6,0xff]
c.fswsp  fa7, 252(sp)
# no-ext-error@-1 {{instruction requires the following: 'C' (Compressed Instructions) and 'F' (Single-Precision Floating-Point) or 'Zcf' (Compressed Single-Precision Floating-Point Instructions)}}
# no-rv32-error@-2 {{instruction requires the following: 'C' (Compressed Instructions) and 'F' (Single-Precision Floating-Point) or 'Zcf' (Compressed Single-Precision Floating-Point Instructions), RV32I Base Instruction Set}}

# CHECK-ASM-AND-OBJ: c.flw  fa3, 124(a5)
# CHECK-ASM: encoding: [0xf4,0x7f]
c.flw  fa3, 124(a5)
# no-ext-error@-1 {{instruction requires the following: 'C' (Compressed Instructions) and 'F' (Single-Precision Floating-Point) or 'Zcf' (Compressed Single-Precision Floating-Point Instructions)}}
# no-rv32-error@-2 {{instruction requires the following: 'C' (Compressed Instructions) and 'F' (Single-Precision Floating-Point) or 'Zcf' (Compressed Single-Precision Floating-Point Instructions), RV32I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: c.fsw  fa2, 124(a1)
# CHECK-ASM: encoding: [0xf0,0xfd]
c.fsw  fa2, 124(a1)
# no-ext-error@-1 {{instruction requires the following: 'C' (Compressed Instructions) and 'F' (Single-Precision Floating-Point) or 'Zcf' (Compressed Single-Precision Floating-Point Instructions)}}
# no-rv32-error@-2 {{instruction requires the following: 'C' (Compressed Instructions) and 'F' (Single-Precision Floating-Point) or 'Zcf' (Compressed Single-Precision Floating-Point Instructions), RV32I Base Instruction Set}}
