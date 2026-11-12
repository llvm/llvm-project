# RUN: llvm-mc %s -triple mips-unknown-linux-gnu -mcpu=mips32r2 \
# RUN:   | FileCheck %s -check-prefix=ASM
#
# RUN: llvm-mc %s -triple mips-unknown-linux-gnu -mcpu=mips32r2 \
# RUN:            -filetype=obj -o - \
# RUN:  | llvm-objdump -d -r - | FileCheck %s --check-prefix=OBJ-O32

# RUN: llvm-mc %s -triple mips64-unknown-linux-gnuabin32 -mcpu=mips64r2 \
# RUN:            -filetype=obj -o - \
# RUN:  | llvm-objdump -d -r - | FileCheck %s --check-prefix=OBJ-N32

# RUN: llvm-mc %s -triple mips64-unknown-linux-gnu -mcpu=mips64r2 \
# RUN:            -filetype=obj -o - \
# RUN:  | llvm-objdump -d -r - | FileCheck %s --check-prefix=OBJ-N64

# ASM:    .text
# ASM:    .option pic2
# ASM:    .set noreorder
# ASM:    .cpload $25
# ASM:    .set reorder

# OBJ-O32:    .text
# OBJ-O32:    lui $gp, 0
# OBJ-O32: R_MIPS_HI16 _gp_disp
# OBJ-O32:    addiu $gp, $gp, 0
# OBJ-O32: R_MIPS_LO16 _gp_disp
# OBJ-O32:    addu $gp, $gp, $25

# OBJ-N32-NOT: .text
# OBJ-N32-NOT: lui   $gp, 0
# OBJ-N32-NOT: addiu $gp, $gp, 0
# OBJ-N32-NOT: addu  $gp, $gp, $25

# OBJ-N64-NOT: .text
# OBJ-N64-NOT: lui   $gp, 0
# OBJ-N64-NOT: addiu $gp, $gp, 0
# OBJ-N64-NOT: addu  $gp, $gp, $25

        .text
        .option pic2
        .set noreorder
        .cpload $25
        .set reorder
