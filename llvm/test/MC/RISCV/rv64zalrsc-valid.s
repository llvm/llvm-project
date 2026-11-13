# RUN: llvm-mc %s -triple=riscv64 -mattr=+a -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+a < %s \
# RUN:     | llvm-objdump --mattr=+a -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify -triple riscv32 -mattr=+a < %s
#
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zalrsc -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zalrsc < %s \
# RUN:     | llvm-objdump --mattr=+zalrsc -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: llvm-mc --verify -triple riscv32 -mattr=+zalrsc < %s

# CHECK-ASM-AND-OBJ: lr.d t0, (t1)
# CHECK-ASM: encoding: [0xaf,0x32,0x03,0x10]
lr.d t0, (t1) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: lr.d.aq t1, (t2)
# CHECK-ASM: encoding: [0x2f,0xb3,0x03,0x14]
lr.d.aq t1, (t2) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: lr.d.rl t2, (t3)
# CHECK-ASM: encoding: [0xaf,0x33,0x0e,0x12]
lr.d.rl t2, (t3) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: lr.d.aqrl t3, (t4)
# CHECK-ASM: encoding: [0x2f,0xbe,0x0e,0x16]
lr.d.aqrl t3, (t4) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}

# CHECK-ASM-AND-OBJ: sc.d t6, t5, (t4)
# CHECK-ASM: encoding: [0xaf,0xbf,0xee,0x19]
sc.d t6, t5, (t4) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: sc.d.aq t5, t4, (t3)
# CHECK-ASM: encoding: [0x2f,0x3f,0xde,0x1d]
sc.d.aq t5, t4, (t3) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: sc.d.rl t4, t3, (t2)
# CHECK-ASM: encoding: [0xaf,0xbe,0xc3,0x1b]
sc.d.rl t4, t3, (t2) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
# CHECK-ASM-AND-OBJ: sc.d.aqrl t3, t2, (t1)
# CHECK-ASM: encoding: [0x2f,0x3e,0x73,0x1e]
sc.d.aqrl t3, t2, (t1) # expected-error:1 {{instruction requires the following: RV64I Base Instruction Set}}
