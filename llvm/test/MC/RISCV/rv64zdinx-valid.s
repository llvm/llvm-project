# RUN: llvm-mc %s -triple=riscv64 -mattr=+zdinx -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zdinx %s \
# RUN:     | llvm-objdump --mattr=+zdinx -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify -triple riscv32 -mattr=+zdinx %s

# CHECK-ASM-AND-OBJ: fcvt.l.d a0, t0, dyn
# CHECK-ASM: encoding: [0x53,0xf5,0x22,0xc2]
fcvt.l.d a0, t0, dyn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.lu.d a1, t1, dyn
# CHECK-ASM: encoding: [0xd3,0x75,0x33,0xc2]
fcvt.lu.d a1, t1, dyn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.d.l t3, a3, dyn
# CHECK-ASM: encoding: [0x53,0xfe,0x26,0xd2]
fcvt.d.l t3, a3, dyn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.d.lu t4, a4, dyn
# CHECK-ASM: encoding: [0xd3,0x7e,0x37,0xd2]
fcvt.d.lu t4, a4, dyn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

# Rounding modes
# CHECK-ASM-AND-OBJ: fcvt.d.l t3, a3, rne
# CHECK-ASM: encoding: [0x53,0x8e,0x26,0xd2]
fcvt.d.l t3, a3, rne # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.d.lu t4, a4, rtz
# CHECK-ASM: encoding: [0xd3,0x1e,0x37,0xd2]
fcvt.d.lu t4, a4, rtz # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.l.d a0, t0, rdn
# CHECK-ASM: encoding: [0x53,0xa5,0x22,0xc2]
fcvt.l.d a0, t0, rdn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.lu.d a1, t1, rup
# CHECK-ASM: encoding: [0xd3,0x35,0x33,0xc2]
fcvt.lu.d a1, t1, rup # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
