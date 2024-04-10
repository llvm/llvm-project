# RUN: llvm-mc %s -triple=riscv32 -mattr=+a -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+a -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+a < %s \
# RUN:     | llvm-objdump --mattr=+a -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+a < %s \
# RUN:     | llvm-objdump --mattr=+a -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-zalrsc -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+experimental-zalrsc -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+experimental-zalrsc < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zalrsc -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+experimental-zalrsc < %s \
# RUN:     | llvm-objdump --mattr=+experimental-zalrsc -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# CHECK-ASM-AND-OBJ: lr.w t0, (t1)
# CHECK-ASM: encoding: [0xaf,0x22,0x03,0x10]
lr.w t0, (t1)
# CHECK-ASM-AND-OBJ: lr.w.aq t1, (t2)
# CHECK-ASM: encoding: [0x2f,0xa3,0x03,0x14]
lr.w.aq t1, (t2)
# CHECK-ASM-AND-OBJ: lr.w.rl t2, (t3)
# CHECK-ASM: encoding: [0xaf,0x23,0x0e,0x12]
lr.w.rl t2, (t3)
# CHECK-ASM-AND-OBJ: lr.w.aqrl t3, (t4)
# CHECK-ASM: encoding: [0x2f,0xae,0x0e,0x16]
lr.w.aqrl t3, (t4)

# CHECK-ASM-AND-OBJ: sc.w t6, t5, (t4)
# CHECK-ASM: encoding: [0xaf,0xaf,0xee,0x19]
sc.w t6, t5, (t4)
# CHECK-ASM-AND-OBJ: sc.w.aq t5, t4, (t3)
# CHECK-ASM: encoding: [0x2f,0x2f,0xde,0x1d]
sc.w.aq t5, t4, (t3)
# CHECK-ASM-AND-OBJ: sc.w.rl t4, t3, (t2)
# CHECK-ASM: encoding: [0xaf,0xae,0xc3,0x1b]
sc.w.rl t4, t3, (t2)
# CHECK-ASM-AND-OBJ: sc.w.aqrl t3, t2, (t1)
# CHECK-ASM: encoding: [0x2f,0x2e,0x73,0x1e]
sc.w.aqrl t3, t2, (t1)
