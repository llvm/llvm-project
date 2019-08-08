# RUN: llvm-mc -filetype=obj -triple mips -mcpu=mips2 %s -o - \
# RUN:   | llvm-objdump -d -r - | FileCheck %s --check-prefix=MIPS
# RUN: llvm-mc -filetype=obj -triple mips -mcpu=mips32 %s -o - \
# RUN:   | llvm-objdump -d -r - | FileCheck %s --check-prefix=MIPS
# RUN: llvm-mc -filetype=obj -triple mips -mcpu=mips32r2 %s -o - \
# RUN:   | llvm-objdump -d -r - | FileCheck %s --check-prefix=MIPS
# RUN: llvm-mc -filetype=obj -triple mips -mcpu=mips3 %s -o - \
# RUN:   | llvm-objdump -d -r - | FileCheck %s --check-prefix=MIPS
# RUN: llvm-mc -filetype=obj -triple mips -mcpu=mips64 %s -o - \
# RUN:   | llvm-objdump -d -r - | FileCheck %s --check-prefix=MIPS
# RUN: llvm-mc -filetype=obj -triple mips -mcpu=mips64r2 %s -o - \
# RUN:   | llvm-objdump -d -r - | FileCheck %s --check-prefix=MIPS
# RUN: llvm-mc -filetype=obj -triple mips -mcpu=mips32r6 %s -o - \
# RUN:   | llvm-objdump -d -r - | FileCheck %s --check-prefix=MIPSR6
# RUN: llvm-mc -filetype=obj -triple mips -mcpu=mips64r6 %s -o - \
# RUN:   | llvm-objdump -d -r - | FileCheck %s --check-prefix=MIPSR6
# RUN: llvm-mc -filetype=obj -triple mips -mcpu=mips32r2 -mattr=+micromips %s -o - \
# RUN:   | llvm-objdump -d -r - | FileCheck %s --check-prefixes=MICROMIPS,MICROMIPSR2
# RUN: llvm-mc -filetype=obj -triple mips -mcpu=mips32r6 -mattr=+micromips %s -o - \
# RUN:   | llvm-objdump -d -r - | FileCheck %s --check-prefixes=MICROMIPS,MICROMIPSR6

# MIPS:         0:  e0 6c 00 00    sc   $12, 0($3)
# MIPSR6:       0:  7c 6c 00 26    sc   $12, 0($3)
# MICROMIPS:    0:  61 83 b0 00    sc   $12, 0($3)
sc $12, 0($3)

# MIPS:         4:  e0 6c 00 04    sc   $12, 4($3)
# MIPSR6:       4:  7c 6c 02 26    sc   $12, 4($3)
# MICROMIPS:    4:  61 83 b0 04    sc   $12, 4($3)
sc $12, 4($3)

# MIPS:          8:  3c 01 00 00    lui  $1, 0
# MIPS:         00000008:  R_MIPS_HI16  symbol
# MIPS:          c:  e0 2c 00 00    sc   $12, 0($1)
# MIPS:         0000000c:  R_MIPS_LO16  symbol

# MIPSR6:        8: 3c 01 00 00     aui    $1, $zero, 0
# MIPSR6:			  00000008:  R_MIPS_HI16	symbol
# MIPSR6:        c: 24 21 00 00     addiu  $1, $1, 0
# MIPSR6:			  0000000c:  R_MIPS_LO16	symbol
# MIPSR6:       10: 7c 2c 00 26     sc     $12, 0($1)

# MICROMIPSR2:   8:  41 a1 00 00    lui  $1, 0
# MICROMIPSR2:  00000008:  R_MICROMIPS_HI16  symbol
# MICROMIPSR2:   c:  61 81 b0 00    sc   $12, 0($1)
# MICROMIPSR2:  0000000c:  R_MICROMIPS_LO16  symbol

# MICROMIPSR6:   8:  3c 01 00 00    lh   $zero, 0($1)
# MICROMIPSR6:  00000008:  R_MICROMIPS_HI16  symbol
# MICROMIPSR6:   c:  61 81 b0 00    sc   $12, 0($1)
# MICROMIPSR6:  0000000c:  R_MICROMIPS_LO16  symbol
sc $12, symbol

# MIPS:         10:  3c 01 00 00    lui  $1, 0
# MIPS:         00000010:  R_MIPS_HI16  symbol
# MIPS:         14:  e0 2c 00 08    sc   $12, 8($1)
# MIPS:         00000014:  R_MIPS_LO16  symbol

# MIPSR6:       14: 3c 01 00 00     aui    $1, $zero, 0
# MIPSR6:       00000014:  R_MIPS_HI16	symbol
# MIPSR6:       18: 24 21 00 08     addiu  $1, $1, 8
# MIPSR6:       00000018:  R_MIPS_LO16	symbol
# MIPSR6:       1c: 7c 2c 00 26     sc     $12, 0($1)

# MICROMIPSR2:  10:  41 a1 00 00    lui  $1, 0
# MICROMIPSR2:  00000010:  R_MICROMIPS_HI16  symbol
# MICROMIPSR2:  14:  61 81 b0 08    sc   $12, 8($1)
# MICROMIPSR2:  00000014:  R_MICROMIPS_LO16  symbol

# MICROMIPSR6:  10:  3c 01 00 00    lh   $zero, 0($1)
# MICROMIPSR6:  00000010:  R_MICROMIPS_HI16  symbol
# MICROMIPSR6:  14:  61 81 b0 08    sc   $12, 8($1)
# MICROMIPSR6:  00000014:  R_MICROMIPS_LO16  symbol
sc $12, symbol + 8
