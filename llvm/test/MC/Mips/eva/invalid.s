# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips32r2 \
# RUN:     -mattr==eva 2>%t1
# RUN: FileCheck %s < %t1

    .set noat
    cachee -1, 255($7) # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
    cachee 32, 255($7) # CHECK: :[[@LINE]]:12: error: invalid operand for instruction
    prefe -1, 255($7)  # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
    prefe 32, 255($7)  # CHECK: :[[@LINE]]:11: error: invalid operand for instruction
    lle $33, 8($5)     # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
    lle $4, 8($33)     # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    lle $4, 512($5)    # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    lle $4, -513($5)   # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    lwe $33, 8($5)     # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
    lwe $4, 8($33)     # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    lwe $4, 512($5)    # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    lwe $4, -513($5)   # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    sbe $33, 8($5)     # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
    sbe $4, 8($33)     # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    sbe $4, 512($5)    # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    sbe $4, -513($5)   # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    sce $33, 8($5)     # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
    sce $4, 8($33)     # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    sce $4, 512($5)    # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    sce $4, -513($5)   # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    she $33, 8($5)     # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
    she $4, 8($33)     # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    she $4, 512($5)    # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    she $4, -513($5)   # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    swe $33, 8($4)     # CHECK: :[[@LINE]]:9: error: invalid operand for instruction
    swe $5, 8($34)     # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    swe $5, 512($4)    # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
    swe $5, -513($4)   # CHECK: :[[@LINE]]:13: error: invalid operand for instruction
