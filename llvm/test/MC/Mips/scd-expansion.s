# RUN: llvm-mc -filetype=obj -triple mips64 -mcpu=mips64 %s -o - \
# RUN:   | llvm-objdump -d -r - | FileCheck %s --check-prefix=MIPS64
# RUN: llvm-mc -filetype=obj -triple mips64 -mcpu=mips64r6 %s -o - \
# RUN:   | llvm-objdump -d -r - | FileCheck %s --check-prefix=MIPS64R6

scd $2, 128($sp)
# MIPS64:         f3 a2 00 80  scd    $2, 128($sp)
# MIPS64R6:       7f a2 40 27  scd    $2, 128($sp)

scd $2, -128($sp)
# MIPS64:         f3 a2 ff 80  scd    $2, -128($sp)
# MIPS64R6:       7f a2 c0 27  scd    $2, -128($sp)

scd $2, 256($sp)
# MIPS64:         f3 a2 01 00  scd    $2, 256($sp)

# MIPS64R6:       67 a1 01 00  daddiu $1, $sp, 256
# MIPS64R6-NEXT:  7c 22 00 27  scd    $2, 0($1)

scd $2, -257($sp)
# MIPS64:         f3 a2 fe ff  scd    $2, -257($sp)

# MIPS64R6:       67 a1 fe ff  daddiu $1, $sp, -257
# MIPS64R6-NEXT:  7c 22 00 27  scd    $2, 0($1)

scd $2, 32767($sp)
# MIPS64:         f3 a2 7f ff  scd    $2, 32767($sp)

# MIPS64R6:       67 a1 7f ff  daddiu $1, $sp, 32767
# MIPS64R6-NEXT:  7c 22 00 27  scd    $2, 0($1)

scd $2, 32768($sp)
# MIPS64:         3c 01 00 01  lui    $1, 1
# MIPS64-NEXT:    00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64-NEXT:    f0 22 80 00  scd    $2, -32768($1)

# MIPS64R6:       34 01 80 00  ori    $1, $zero, 32768
# MIPS64R6-NEXT:  00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64R6-NEXT:  7c 22 00 27  scd    $2, 0($1)

scd $2, -32768($sp)
# MIPS64:         f3 a2 80 00  scd    $2, -32768($sp)

# MIPS64R6:       67 a1 80 00  daddiu $1, $sp, -32768
# MIPS64R6-NEXT:  7c 22 00 27  scd    $2, 0($1)

scd $2, -32769($sp)
# MIPS64:         3c 01 ff ff  lui    $1, 65535
# MIPS64-NEXT:    00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64-NEXT:    f0 22 7f ff  scd    $2, 32767($1)

# MIPS64R6:       3c 01 ff ff  aui    $1, $zero, 65535
# MIPS64R6-NEXT:  34 21 7f ff  ori    $1, $1, 32767
# MIPS64R6-NEXT:  00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64R6-NEXT:  7c 22 00 27  scd    $2, 0($1)

scd $2, 2147483648($sp)
# MIPS64:         34 01 80 00  ori    $1, $zero, 32768
# MIPS64-NEXT:    00 01 0c 38  dsll   $1, $1, 16
# MIPS64-NEXT:    00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64-NEXT:    f0 22 00 00  scd    $2, 0($1)

# MIPS64R6:       34 01 80 00  ori    $1, $zero, 32768
# MIPS64R6-NEXT:  00 01 0c 38  dsll   $1, $1, 16
# MIPS64R6-NEXT:  00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64R6-NEXT:  7c 22 00 27  scd    $2, 0($1)

scd $2, -2147483648($sp)
# MIPS64:         3c 01 80 00  lui    $1, 32768
# MIPS64-NEXT:    00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64-NEXT:    f0 22 00 00  scd    $2, 0($1)

# MIPS64R6:       3c 01 80 00  aui    $1, $zero, 32768
# MIPS64R6-NEXT:  00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64R6-NEXT:  7c 22 00 27  scd    $2, 0($1)

scd $2, 9223372036853775808($sp)
# MIPS64:         3c 01 7f ff  lui    $1, 32767
# MIPS64-NEXT:    34 21 ff ff  ori    $1, $1, 65535
# MIPS64-NEXT:    00 01 0c 38  dsll   $1, $1, 16
# MIPS64-NEXT:    34 21 ff f1  ori    $1, $1, 65521
# MIPS64-NEXT:    00 01 0c 38  dsll   $1, $1, 16
# MIPS64-NEXT:    00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64-NEXT:    f0 22 bd c0  scd    $2, -16960($1)

# MIPS64R6:       3c 01 7f ff  aui    $1, $zero, 32767
# MIPS64R6-NEXT:  34 21 ff ff  ori    $1, $1, 65535
# MIPS64R6-NEXT:  00 01 0c 38  dsll   $1, $1, 16
# MIPS64R6-NEXT:  34 21 ff f0  ori    $1, $1, 65520
# MIPS64R6-NEXT:  00 01 0c 38  dsll   $1, $1, 16
# MIPS64R6-NEXT:  34 21 bd c0  ori    $1, $1, 48576
# MIPS64R6-NEXT:  00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64R6-NEXT:  7c 22 00 27  scd    $2, 0($1)

scd $12, symbol
# MIPS64:         3c 01 00 00  lui    $1, 0
# MIPS64-NEXT:               R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    64 21 00 00  daddiu $1, $1, 0
# MIPS64-NEXT:               R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    00 01 0c 38  dsll   $1, $1, 16
# MIPS64-NEXT:    64 21 00 00  daddiu $1, $1, 0
# MIPS64-NEXT:               R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    00 01 0c 38  dsll   $1, $1, 16
# MIPS64-NEXT:    f0 2c 00 00  scd    $12, 0($1)
# MIPS64-NEXT:               R_MIPS_LO16/R_MIPS_NONE/R_MIPS_NONE  symbol

# MIPS64R6:       3c 01 00 00  aui    $1, $zero, 0
# MIPS64R6-NEXT:             R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64R6-NEXT:  64 21 00 00  daddiu $1, $1, 0
# MIPS64R6-NEXT:             R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64R6-NEXT:  00 01 0c 38  dsll   $1, $1, 16
# MIPS64R6-NEXT:  64 21 00 00  daddiu $1, $1, 0
# MIPS64R6-NEXT:             R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64R6-NEXT:  00 01 0c 38  dsll   $1, $1, 16
# MIPS64R6-NEXT:  64 21 00 00  daddiu $1, $1, 0
# MIPS64R6-NEXT:             R_MIPS_LO16/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64R6-NEXT:  7c 2c 00 27  scd    $12, 0($1)

scd $12, symbol($3)
# MIPS64:         3c 01 00 00  lui    $1, 0
# MIPS64-NEXT:               R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    64 21 00 00  daddiu $1, $1, 0
# MIPS64-NEXT:               R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    00 01 0c 38  dsll   $1, $1, 16
# MIPS64-NEXT:    64 21 00 00  daddiu $1, $1, 0
# MIPS64-NEXT:               R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    00 01 0c 38  dsll   $1, $1, 16
# MIPS64-NEXT:    00 23 08 2d  daddu  $1, $1, $3
# MIPS64-NEXT:    f0 2c 00 00  scd    $12, 0($1)
# MIPS64-NEXT:               R_MIPS_LO16/R_MIPS_NONE/R_MIPS_NONE  symbol

# MIPS64R6:       3c 01 00 00  aui    $1, $zero, 0
# MIPS64R6-NEXT:             R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64R6-NEXT:  64 21 00 00  daddiu $1, $1, 0
# MIPS64R6-NEXT:             R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64R6-NEXT:  00 01 0c 38  dsll   $1, $1, 16
# MIPS64R6-NEXT:  64 21 00 00  daddiu $1, $1, 0
# MIPS64R6-NEXT:             R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64R6-NEXT:  00 01 0c 38  dsll   $1, $1, 16
# MIPS64R6-NEXT:  64 21 00 00  daddiu $1, $1, 0
# MIPS64R6-NEXT:             R_MIPS_LO16/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64R6-NEXT:  00 23 08 2d  daddu  $1, $1, $3
# MIPS64R6-NEXT:  7c 2c 00 27  scd    $12, 0($1)

scd $12, symbol+8
# MIPS64:         3c 01 00 00  lui    $1, 0
# MIPS64-NEXT:               R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64-NEXT:    64 21 00 00  daddiu $1, $1, 0
# MIPS64-NEXT:               R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64-NEXT:    00 01 0c 38  dsll   $1, $1, 16
# MIPS64-NEXT:    64 21 00 00  daddiu $1, $1, 0
# MIPS64-NEXT:               R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64-NEXT:    00 01 0c 38  dsll   $1, $1, 16
# MIPS64-NEXT:    f0 2c 00 00  scd    $12, 0($1)
# MIPS64-NEXT:               R_MIPS_LO16/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8

# MIPS64R6:       3c 01 00 00  aui    $1, $zero, 0
# MIPS64R6-NEXT:             R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64R6-NEXT:  64 21 00 00  daddiu $1, $1, 0
# MIPS64R6-NEXT:             R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64R6-NEXT:  00 01 0c 38  dsll   $1, $1, 16
# MIPS64R6-NEXT:  64 21 00 00  daddiu $1, $1, 0
# MIPS64R6-NEXT:             R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64R6-NEXT:  00 01 0c 38  dsll   $1, $1, 16
# MIPS64R6-NEXT:  64 21 00 00  daddiu $1, $1, 0
# MIPS64R6-NEXT:             R_MIPS_LO16/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64R6-NEXT:  7c 2c 00 27  scd    $12, 0($1)

.option pic2

scd $12, symbol
# MIPS64:         df 81 00 00  ld     $1, 0($gp)
# MIPS64-NEXT:               R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE symbol
# MIPS64-NEXT:    f0 2c 00 00  scd    $12, 0($1)

# MIPS64R6:       df 81 00 00  ld     $1, 0($gp)
# MIPS64R6-NEXT:             R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE symbol
# MIPS64R6-NEXT:  7c 2c 00 27  scd    $12, 0($1)

scd $12, symbol+8
# MIPS64:         df 81 00 00  ld     $1, 0($gp)
# MIPS64-NEXT:               R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE symbol
# MIPS64-NEXT:    f0 2c 00 08  scd    $12, 8($1)

# MIPS64R6:       df 81 00 00  ld     $1, 0($gp)
# MIPS64R6-NEXT:             R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE symbol
# MIPS64R6-NEXT:  64 21 00 08  daddiu $1, $1, 8
# MIPS64R6-NEXT:  7c 2c 00 27  scd    $12, 0($1)
