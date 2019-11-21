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

# MIPS:         e0 6c 00 00    sc   $12, 0($3)
# MIPSR6:       7c 6c 00 26    sc   $12, 0($3)
# MICROMIPS:    61 83 b0 00    sc   $12, 0($3)
sc $12, 0($3)

# MIPS:         e0 6c 00 04    sc   $12, 4($3)
# MIPSR6:       7c 6c 02 26    sc   $12, 4($3)
# MICROMIPS:    61 83 b0 04    sc   $12, 4($3)
sc $12, 4($3)

# MIPS:         3c 01 00 00    lui  $1, 0
# MIPS:                    R_MIPS_HI16  symbol
# MIPS:         e0 2c 00 00    sc   $12, 0($1)
# MIPS:                    R_MIPS_LO16  symbol

# MIPSR6:       3c 01 00 00     aui    $1, $zero, 0
# MIPSR6:			             R_MIPS_HI16	symbol
# MIPSR6:       24 21 00 00     addiu  $1, $1, 0
# MIPSR6:			             R_MIPS_LO16	symbol
# MIPSR6:       7c 2c 00 26     sc     $12, 0($1)

# MICROMIPSR2:  41 a1 00 00    lui  $1, 0
# MICROMIPSR2:             R_MICROMIPS_HI16  symbol
# MICROMIPSR2:  61 81 b0 00    sc   $12, 0($1)
# MICROMIPSR2:             R_MICROMIPS_LO16  symbol

# MICROMIPSR6:  3c 01 00 00    lh   $zero, 0($1)
# MICROMIPSR6:             R_MICROMIPS_HI16  symbol
# MICROMIPSR6:  61 81 b0 00    sc   $12, 0($1)
# MICROMIPSR6:             R_MICROMIPS_LO16  symbol
sc $12, symbol

# MIPS:         3c 01 00 00    lui  $1, 0
# MIPS:                    R_MIPS_HI16  symbol
# MIPS:         e0 2c 00 08    sc   $12, 8($1)
# MIPS:                    R_MIPS_LO16  symbol

# MIPSR6:       3c 01 00 00     aui    $1, $zero, 0
# MIPSR6:                  R_MIPS_HI16	symbol
# MIPSR6:       24 21 00 08     addiu  $1, $1, 8
# MIPSR6:                  R_MIPS_LO16	symbol
# MIPSR6:       7c 2c 00 26     sc     $12, 0($1)

# MICROMIPSR2:  41 a1 00 00    lui  $1, 0
# MICROMIPSR2:             R_MICROMIPS_HI16  symbol
# MICROMIPSR2:  61 81 b0 08    sc   $12, 8($1)
# MICROMIPSR2:             R_MICROMIPS_LO16  symbol

# MICROMIPSR6:  3c 01 00 00    lh   $zero, 0($1)
# MICROMIPSR6:             R_MICROMIPS_HI16  symbol
# MICROMIPSR6:  61 81 b0 08    sc   $12, 8($1)
# MICROMIPSR6:             R_MICROMIPS_LO16  symbol
sc $12, symbol + 8
