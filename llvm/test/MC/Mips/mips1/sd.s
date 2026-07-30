# RUN: llvm-mc -filetype=obj -triple mips   -mcpu=mips1 %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d - | FileCheck %s --check-prefix=MIPS1-EB
# RUN: llvm-mc -filetype=obj -triple mipsel -mcpu=mips1 %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d - | FileCheck %s --check-prefix=MIPS1-EL

# Check if s.d instruction alias is suported on Mips1.

# MIPS1-EB:    0: e4 c1 00 00    swc1    $f1, 0($6)
# MIPS1-EB:    4: e4 c0 00 04    swc1    $f0, 4($6)

# MIPS1-EL:    0: 00 00 c0 e4    swc1    $f0, 0($6)
# MIPS1-EL:    4: 04 00 c1 e4    swc1    $f1, 4($6)
s.d $f0, 0($6)
