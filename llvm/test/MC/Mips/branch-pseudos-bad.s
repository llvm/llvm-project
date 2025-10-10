# RUN: not llvm-mc %s -triple=mips -mcpu=mips32 2>&1 | FileCheck %s

# CHECK: error: invalid operand for instruction
  beql $t0, ($t0), 1
# CHECK: error: invalid operand for instruction
  bne $t0, ($t0), 1
# CHECK: error: invalid operand for instruction
  beq $t0, ($t0), 1


# Check for errors when using conditional branch pseudos after .set noat.
  .set noat
local_label:
  blt $7, $8, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
  bltu $7, $8, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
  ble $7, $8, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
  bleu $7, $8, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
  bge $7, $8, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
  bgeu $7, $8, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
  bgt $7, $8, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
  bgtu $7, $8, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available

  beql $7, 256, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
  bnel $7, 256, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
  bltl $7, $8, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
  bltul $7, $8, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
  blel $7, $8, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
  bleul $7, $8, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
  bgel $7, $8, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
  bgeul $7, $8, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
  bgtl $7, $8, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
  bgtul $7, $8, local_label
# CHECK: :[[@LINE-1]]:3: error: pseudo-instruction requires $at, which is not available
