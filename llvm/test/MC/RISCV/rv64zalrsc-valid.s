# RUN: llvm-mc %s -triple=riscv64 -mattr=+a -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+a < %s \
# RUN:     | llvm-objdump --mattr=+a -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 -mattr=+a < %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-RV32 %s
#
# RUN: llvm-mc %s -triple=riscv64 -mattr=+zalrsc -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+zalrsc < %s \
# RUN:     | llvm-objdump --mattr=+zalrsc -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
#
# RUN: not llvm-mc -triple riscv32 -mattr=+zalrsc < %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-RV32 %s

# CHECK-ASM-AND-OBJ: lr.d t0, (t1)
# CHECK-ASM: encoding: [0xaf,0x32,0x03,0x10]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
lr.d t0, (t1)
# CHECK-ASM-AND-OBJ: lr.d.aq t1, (t2)
# CHECK-ASM: encoding: [0x2f,0xb3,0x03,0x14]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
lr.d.aq t1, (t2)
# CHECK-ASM-AND-OBJ: lr.d.rl t2, (t3)
# CHECK-ASM: encoding: [0xaf,0x33,0x0e,0x12]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
lr.d.rl t2, (t3)
# CHECK-ASM-AND-OBJ: lr.d.aqrl t3, (t4)
# CHECK-ASM: encoding: [0x2f,0xbe,0x0e,0x16]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
lr.d.aqrl t3, (t4)

# CHECK-ASM-AND-OBJ: sc.d t6, t5, (t4)
# CHECK-ASM: encoding: [0xaf,0xbf,0xee,0x19]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
sc.d t6, t5, (t4)
# CHECK-ASM-AND-OBJ: sc.d.aq t5, t4, (t3)
# CHECK-ASM: encoding: [0x2f,0x3f,0xde,0x1d]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
sc.d.aq t5, t4, (t3)
# CHECK-ASM-AND-OBJ: sc.d.rl t4, t3, (t2)
# CHECK-ASM: encoding: [0xaf,0xbe,0xc3,0x1b]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
sc.d.rl t4, t3, (t2)
# CHECK-ASM-AND-OBJ: sc.d.aqrl t3, t2, (t1)
# CHECK-ASM: encoding: [0x2f,0x3e,0x73,0x1e]
# CHECK-RV32: :[[@LINE+1]]:1: error: instruction requires the following: RV64I Base Instruction Set{{$}}
sc.d.aqrl t3, t2, (t1)
