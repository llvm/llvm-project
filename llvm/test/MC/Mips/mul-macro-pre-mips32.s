# Before MIPS32 there is no mul instruction, so the register forms of mul are
# expanded to multu and mflo, as GAS does.

# RUN: llvm-mc %s -triple=mips -mcpu=mips1 -show-encoding | FileCheck %s
# RUN: llvm-mc %s -triple=mips -mcpu=mips2 -show-encoding | FileCheck %s
# RUN: llvm-mc %s -triple=mips64 -mcpu=mips3 -show-encoding | FileCheck %s
# RUN: llvm-mc %s -triple=mips64 -mcpu=mips4 -show-encoding | FileCheck %s
# RUN: llvm-mc %s -triple=mips64 -mcpu=mips5 -show-encoding | FileCheck %s
# RUN: llvm-mc %s -triple=mipsel -mcpu=r5900 | FileCheck %s --check-prefix=R5900

  mul $4, $5, $6
# CHECK:      multu $5, $6           # encoding: [0x00,0xa6,0x00,0x19]
# CHECK-NEXT: mflo  $4               # encoding: [0x00,0x00,0x20,0x12]
# R5900:      multu $5, $6
# R5900-NEXT: mflo  $4

  mul $4, $5
# CHECK:      multu $4, $5           # encoding: [0x00,0x85,0x00,0x19]
# CHECK-NEXT: mflo  $4               # encoding: [0x00,0x00,0x20,0x12]
# R5900:      multu $4, $5
# R5900-NEXT: mflo  $4

  mul $4, $5, 10
# CHECK:      addiu $1, $zero, 10    # encoding: [0x24,0x01,0x00,0x0a]
# CHECK-NEXT: mult  $5, $1           # encoding: [0x00,0xa1,0x00,0x18]
# CHECK-NEXT: mflo  $4               # encoding: [0x00,0x00,0x20,0x12]
# R5900:      addiu $1, $zero, 10
# R5900-NEXT: mult  $5, $1
# R5900-NEXT: mflo  $4
