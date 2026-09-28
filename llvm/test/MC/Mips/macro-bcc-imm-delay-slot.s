# RUN: llvm-mc %s -triple=mipsel -mcpu=mips32r2 -filetype=obj -o - | \
# RUN:   llvm-objdump -d -z - | FileCheck %s

.text
.set reorder
reorder:
  blt $9, -1022, target
  bne $25, $zero, target
  nop

.set noreorder
noreorder:
  blt $9, -1022, target
  bne $25, $zero, target
  nop

target:
  nop

# CHECK-LABEL: <reorder>:
# CHECK-NEXT:    addiu $1, $zero, -0x3fe
# CHECK-NEXT:    slt $1, $9, $1
# CHECK-NEXT:    bnez $1,
# CHECK-NEXT:    nop
# CHECK-NEXT:    bnez $25,
# CHECK-NEXT:    nop

# CHECK-LABEL: <noreorder>:
# CHECK-NEXT:    addiu $1, $zero, -0x3fe
# CHECK-NEXT:    slt $1, $9, $1
# CHECK-NEXT:    bnez $1,
# CHECK-NEXT:    bnez $25,
# CHECK-NEXT:    nop
