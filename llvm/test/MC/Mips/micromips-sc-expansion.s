# RUN: llvm-mc -filetype=obj -triple mips -mcpu=mips32r2 -mattr=+micromips %s -o - \
# RUN:   | llvm-objdump -d -r - | FileCheck %s --check-prefix=MICROMIPSR2
# RUN: llvm-mc -filetype=obj -triple mips -mcpu=mips32r6 -mattr=+micromips %s -o - \
# RUN:   | llvm-objdump -d -r - | FileCheck %s --check-prefix=MICROMIPSR6

# MICROMIPSR2:  61 83 b0 00    sc   $12, 0($3)
# MICROMIPSR6:  61 83 b0 00    sc   $12, 0($3)
sc $12, 0($3)

# MICROMIPSR2:  61 83 b0 04    sc   $12, 4($3)
# MICROMIPSR6:  61 83 b0 04    sc   $12, 4($3)
sc $12, 4($3)

# MICROMIPSR2:  41 a1 00 00    lui  $1, 0
# MICROMIPSR2:             R_MICROMIPS_HI16  symbol
# MICROMIPSR2:  61 81 b0 00    sc   $12, 0($1)
# MICROMIPSR2:             R_MICROMIPS_LO16  symbol

# MICROMIPSR6:  3c 01 00 00    lh   $zero, 0($1)
# MICROMIPSR6:             R_MICROMIPS_HI16  symbol
# MICROMIPSR6:  61 81 b0 00    sc   $12, 0($1)
# MICROMIPSR6:             R_MICROMIPS_LO16  symbol
sc $12, symbol

# MICROMIPSR2:  41 a1 00 00    lui  $1, 0
# MICROMIPSR2:             R_MICROMIPS_HI16  symbol
# MICROMIPSR2:  61 81 b0 08    sc   $12, 8($1)
# MICROMIPSR2:             R_MICROMIPS_LO16  symbol

# MICROMIPSR6:  3c 01 00 00    lh   $zero, 0($1)
# MICROMIPSR6:             R_MICROMIPS_HI16  symbol
# MICROMIPSR6:  61 81 b0 08    sc   $12, 8($1)
# MICROMIPSR6:             R_MICROMIPS_LO16  symbol
sc $12, symbol + 8
