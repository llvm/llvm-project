# RUN: llvm-mc %s -triple=riscv64 -mattr=+q -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+q < %s \
# RUN:     | llvm-objdump --mattr=+q -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify -triple riscv32 -mattr=+q < %s

# CHECK-ASM-AND-OBJ: fcvt.l.q a0, ft0, dyn
# CHECK-ASM: encoding: [0x53,0x75,0x20,0xc6]
fcvt.l.q a0, ft0, dyn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.lu.q a1, ft1, dyn
# CHECK-ASM: encoding: [0xd3,0xf5,0x30,0xc6]
fcvt.lu.q a1, ft1, dyn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.q.l ft3, a3, dyn
# CHECK-ASM: encoding: [0xd3,0xf1,0x26,0xd6]
fcvt.q.l ft3, a3, dyn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.q.lu ft4, a4, dyn
# CHECK-ASM: encoding: [0x53,0x72,0x37,0xd6]
fcvt.q.lu ft4, a4, dyn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

# Rounding modes
# CHECK-ASM-AND-OBJ: fcvt.q.l ft3, a3
# CHECK-ASM: encoding: [0xd3,0x81,0x26,0xd6]
fcvt.q.l ft3, a3, rne # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.q.lu ft4, a4, rtz
# CHECK-ASM: encoding: [0x53,0x12,0x37,0xd6]
fcvt.q.lu ft4, a4, rtz # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.l.q a0, ft0, rdn
# CHECK-ASM: encoding: [0x53,0x25,0x20,0xc6]
fcvt.l.q a0, ft0, rdn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.lu.q a1, ft1, rup
# CHECK-ASM: encoding: [0xd3,0xb5,0x30,0xc6]
fcvt.lu.q a1, ft1, rup # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
