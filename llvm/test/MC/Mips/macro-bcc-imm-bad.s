# RUN: not llvm-mc %s -triple=mips -mcpu=mips32r2 2>&1 | \
# RUN:     FileCheck %s --check-prefix=ALL

    .text
    .set noat
foo:
    blt $a2, 16, foo # ALL: :[[@LINE]]:5: error: pseudo-instruction requires $at, which is not available
    .set at
    .set noreorder
    .set nomacro
    blt $a2, 16, foo # ALL: :[[@LINE]]:5: warning: macro instruction expanded into multiple instructions
                     # ALL-NOT: :[[@LINE-1]]:5: warning: macro instruction expanded into multiple instructions
