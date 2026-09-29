# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 | FileCheck %s

# CHECK:    .set msa
# CHECK:    addvi.b     $w14, $w12, 14
# CHECK:    addvi.h     $w26, $w17, 4
# CHECK:    addvi.w     $w19, $w13, 11
# CHECK:    addvi.d     $w16, $w19, 7    
# CHECK:    subvi.b     $w14, $w12, 14
# CHECK:    subvi.h     $w26, $w17, 4
# CHECK:    subvi.w     $w19, $w13, 11
# CHECK:    subvi.d     $w16, $w19, 7

    .set msa
    addvi.b     $w14, $w12, 14
    addvi.h     $w26, $w17, 4
    addvi.w     $w19, $w13, 11
    addvi.d     $w16, $w19, 7
    
    subvi.b     $w14, $w12, 14
    subvi.h     $w26, $w17, 4
    subvi.w     $w19, $w13, 11
    subvi.d     $w16, $w19, 7
