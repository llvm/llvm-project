# RUN: llvm-mc %s -triple=riscv64 -mattr=+zhinx -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zhinx %s \
# RUN:     | llvm-objdump --mattr=+zhinx -M no-aliases -d -r - \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify -triple riscv32 -mattr=+zhinx %s

# CHECK-ASM-AND-OBJ: fcvt.l.h a0, t0, dyn
# CHECK-ASM: encoding: [0x53,0xf5,0x22,0xc4]
fcvt.l.h a0, t0, dyn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.lu.h a1, t1, dyn
# CHECK-ASM: encoding: [0xd3,0x75,0x33,0xc4]
fcvt.lu.h a1, t1, dyn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.h.l t2, a2, dyn
# CHECK-ASM: encoding: [0xd3,0x73,0x26,0xd4]
fcvt.h.l t2, a2, dyn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.h.lu t3, a3, dyn
# CHECK-ASM: encoding: [0x53,0xfe,0x36,0xd4]
fcvt.h.lu t3, a3, dyn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

# Rounding modes
# CHECK-ASM-AND-OBJ: fcvt.l.h a4, t4, rne
# CHECK-ASM: encoding: [0x53,0x87,0x2e,0xc4]
fcvt.l.h a4, t4, rne # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.lu.h a5, t5, rtz
# CHECK-ASM: encoding: [0xd3,0x17,0x3f,0xc4]
fcvt.lu.h a5, t5, rtz # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.h.l t6, a6, rdn
# CHECK-ASM: encoding: [0xd3,0x2f,0x28,0xd4]
fcvt.h.l t6, a6, rdn # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: fcvt.h.lu s7, a7, rup
# CHECK-ASM: encoding: [0xd3,0xbb,0x38,0xd4]
fcvt.h.lu s7, a7, rup # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
