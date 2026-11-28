# RUN: llvm-mc %s -triple=mips -mcpu=mips1 | FileCheck %s --check-prefixes=ALL,EB
# RUN: llvm-mc %s -triple=mipsel -mcpu=mips1 | FileCheck %s --check-prefixes=ALL,EL
# RUN: llvm-mc %s -triple=mips -mcpu=mips2 | FileCheck %s --check-prefix=MIPS2

# MIPS I has no ldc1/sdc1. l.d, s.d and li.d use two lwc1/swc1.

        .text
        l.s     $f0, 4($4)
# ALL:          lwc1    $f0, 4($4)
# MIPS2:        lwc1    $f0, 4($4)

        s.s     $f0, 8($4)
# ALL:          swc1    $f0, 8($4)
# MIPS2:        swc1    $f0, 8($4)

        l.d     $f2, 16($4)
# EB:           lwc1    $f3, 16($4)
# EB-NEXT:      lwc1    $f2, 20($4)
# EL:           lwc1    $f2, 16($4)
# EL-NEXT:      lwc1    $f3, 20($4)
# MIPS2:        ldc1    $f2, 16($4)

        s.d     $f6, 32($4)
# EB:           swc1    $f7, 32($4)
# EB-NEXT:      swc1    $f6, 36($4)
# EL:           swc1    $f6, 32($4)
# EL-NEXT:      swc1    $f7, 36($4)
# MIPS2:        sdc1    $f6, 32($4)

        li.d    $f8, 1.1
# ALL:        [[LABEL:\$tmp[0-9]+]]:
# ALL-NEXT:     .p2align 3
# ALL-NEXT:     .8byte 4607632778762754458
# ALL:          lui     $1, %hi([[LABEL]])
# EB-NEXT:      lwc1    $f9, %lo([[LABEL]])($1)
# EB-NEXT:      lwc1    $f8, %lo([[LABEL]]+4)($1)
# EL-NEXT:      lwc1    $f8, %lo([[LABEL]])($1)
# EL-NEXT:      lwc1    $f9, %lo([[LABEL]]+4)($1)
# MIPS2:      [[LABEL:\$tmp[0-9]+]]:
# MIPS2-NEXT:   .p2align 3
# MIPS2-NEXT:   .8byte 4607632778762754458
# MIPS2:        lui     $1, %hi([[LABEL]])
# MIPS2-NEXT:   ldc1    $f8, %lo([[LABEL]])($1)
