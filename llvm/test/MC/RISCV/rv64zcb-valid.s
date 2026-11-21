# RUN: llvm-mc %s -triple=riscv64 -mattr=+m,+zbb,+zba,+zcb -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+m,+zbb,+zba,+zcb < %s \
# RUN:     | llvm-objdump --mattr=+m,+zbb,+zba,zcb -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify=no-ext -triple riscv64 -M no-aliases -show-encoding < %s
# RUN: llvm-mc --verify=no-rv64 -triple riscv32 -mattr=+m,+zbb,+zba,+zcb -M no-aliases -show-encoding < %s

# CHECK-ASM-AND-OBJ: c.zext.w s0
# CHECK-ASM: encoding: [0x71,0x9c]
c.zext.w s0
# no-ext-error@-1 {{instruction requires the following: 'Zba' (Address Generation Instructions), 'Zcb' (Compressed basic bit manipulation instructions)}}
# no-rv64-error@-2 {{instruction requires the following: RV64I Base Instruction Set}}

# CHECK-ASM-AND-OBJ: c.zext.w s0
# CHECK-ASM: encoding: [0x71,0x9c]
add.uw s0, s0, zero
# no-ext-error@-1 {{instruction requires the following: 'Zba' (Address Generation Instructions)}}
# no-rv64-error@-2 {{instruction requires the following: RV64I Base Instruction Set}}
