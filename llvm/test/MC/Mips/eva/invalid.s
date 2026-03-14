# Instructions that are invalid
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips32r2 \
# RUN:     -mattr=+eva 2>%t1
# RUN: FileCheck %s < %t1

    .set noat
    cachee -1, 255($7) # CHECK: :[[@LINE]]:12: error: expected 5-bit unsigned immediate
    cachee 32, 255($7) # CHECK: :[[@LINE]]:12: error: expected 5-bit unsigned immediate
    prefe -1, 255($7)  # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
    prefe 32, 255($7)  # CHECK: :[[@LINE]]:11: error: expected 5-bit unsigned immediate
    lle $33, 8($5)     # CHECK: :[[@LINE]]:9: error: invalid register number
    lle $4, 8($33)     # CHECK: :[[@LINE]]:15: error: invalid register number
    lle $4, 512($5)    # CHECK: :[[@LINE]]:13: error: expected memory with 9-bit signed offset
    lle $4, -513($5)   # CHECK: :[[@LINE]]:13: error: expected memory with 9-bit signed offset
    lwe $33, 8($5)     # CHECK: :[[@LINE]]:9: error: invalid register number
    lwe $4, 8($33)     # CHECK: :[[@LINE]]:15: error: invalid register number
    lwe $4, 512($5)    # CHECK: :[[@LINE]]:13: error: expected memory with 9-bit signed offset
    lwe $4, -513($5)   # CHECK: :[[@LINE]]:13: error: expected memory with 9-bit signed offset
    sbe $33, 8($5)     # CHECK: :[[@LINE]]:9: error: invalid register number
    sbe $4, 8($33)     # CHECK: :[[@LINE]]:15: error: invalid register number
    sbe $4, 512($5)    # CHECK: :[[@LINE]]:13: error: expected memory with 9-bit signed offset
    sbe $4, -513($5)   # CHECK: :[[@LINE]]:13: error: expected memory with 9-bit signed offset
    sce $33, 8($5)     # CHECK: :[[@LINE]]:9: error: invalid register number
    sce $4, 8($33)     # CHECK: :[[@LINE]]:15: error: invalid register number
    sce $4, 512($5)    # CHECK: :[[@LINE]]:13: error: expected memory with 9-bit signed offset
    sce $4, -513($5)   # CHECK: :[[@LINE]]:13: error: expected memory with 9-bit signed offset
    she $33, 8($5)     # CHECK: :[[@LINE]]:9: error: invalid register number
    she $4, 8($33)     # CHECK: :[[@LINE]]:15: error: invalid register number
    she $4, 512($5)    # CHECK: :[[@LINE]]:13: error: expected memory with 9-bit signed offset
    she $4, -513($5)   # CHECK: :[[@LINE]]:13: error: expected memory with 9-bit signed offset
    swe $33, 8($4)     # CHECK: :[[@LINE]]:9: error: invalid register number
    swe $5, 8($34)     # CHECK: :[[@LINE]]:15: error: invalid register number
    swe $5, 512($4)    # CHECK: :[[@LINE]]:13: error: expected memory with 9-bit signed offset
    swe $5, -513($4)   # CHECK: :[[@LINE]]:13: error: expected memory with 9-bit signed offset

