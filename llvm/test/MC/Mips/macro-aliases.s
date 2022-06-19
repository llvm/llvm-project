# RUN: llvm-mc -arch=mips -mcpu=mips32r2 %s -show-inst | FileCheck %s

# Test that subu accepts constant operands and inverts them when
# rendering the operand.

  subu  $4, $4, 4          # CHECK: ADDiu
                           # CHECK: Imm:-4
  subu  $gp, $gp, 4        # CHECK: ADDiu
                           # CHECK: Imm:-4
  subu  $sp, $sp, 4        # CHECK: ADDiu
                           # CHECK: Imm:-4
  subu  $4, $4, -4         # CHECK: ADDiu
                           # CHECK: Imm:4
  subu  $gp, $gp, -4       # CHECK: ADDiu
                           # CHECK: Imm:4
  subu  $sp, $sp, -4       # CHECK: ADDiu
                           # CHECK: Imm:4
  subu  $sp, $sp, -(4 + 4) # CHECK: ADDiu
                           # CHECK: Imm:8

  subu  $4, 8              # CHECK: ADDiu
                           # CHECK: Imm:-8
  subu  $gp, 8             # CHECK: ADDiu
                           # CHECK: Imm:-8
  subu  $sp, 8             # CHECK: ADDiu
                           # CHECK: Imm:-8
  subu  $4, -8             # CHECK: ADDiu
                           # CHECK: Imm:8
  subu  $gp, -8            # CHECK: ADDiu
                           # CHECK: Imm:8
  subu  $sp, -8            # CHECK: ADDiu
                           # CHECK: Imm:8
  subu  $sp, -(4 + 4)      # CHECK: ADDiu
                           # CHECK: Imm:8

