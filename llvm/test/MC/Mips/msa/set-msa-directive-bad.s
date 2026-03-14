# RUN: not llvm-mc %s -triple=mips -mcpu=mips32r2 2>%t1
# RUN: FileCheck %s < %t1

    .set nomsa
    addvi.b     $w14, $w12, 14 # CHECK: error: instruction requires a CPU feature not currently enabled

    .set msa
    addvi.h     $w26, $w17, 4 
    
    .set nomsa
    addvi.w     $w19, $w13, 11 # CHECK: error: instruction requires a CPU feature not currently enabled
