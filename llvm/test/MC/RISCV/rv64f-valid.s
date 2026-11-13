# RUN: llvm-mc %s -triple=riscv64 -mattr=+f -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+f < %s \
# RUN:     | llvm-objdump --mattr=+f -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify -triple riscv32 -mattr=+f < %s

# CHECK-ASM-AND-OBJ: fcvt.l.s a0, ft0, dyn
# CHECK-ASM: encoding: [0x53,0x75,0x20,0xc0]
fcvt.l.s a0, ft0, dyn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.lu.s a1, ft1, dyn
# CHECK-ASM: encoding: [0xd3,0xf5,0x30,0xc0]
fcvt.lu.s a1, ft1, dyn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.s.l ft2, a2, dyn
# CHECK-ASM: encoding: [0x53,0x71,0x26,0xd0]
fcvt.s.l ft2, a2, dyn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.s.lu ft3, a3, dyn
# CHECK-ASM: encoding: [0xd3,0xf1,0x36,0xd0]
fcvt.s.lu ft3, a3, dyn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

# Rounding modes
# CHECK-ASM-AND-OBJ: fcvt.l.s a4, ft4, rne
fcvt.l.s a4, ft4, rne # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.lu.s a5, ft5, rtz
fcvt.lu.s a5, ft5, rtz # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.s.l ft6, a6, rdn
fcvt.s.l ft6, a6, rdn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.s.lu ft7, a7, rup
fcvt.s.lu ft7, a7, rup # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
