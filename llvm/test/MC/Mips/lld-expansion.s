# RUN: llvm-mc -filetype=obj -triple mips64 -mcpu=mips64 %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s --check-prefix=MIPS64
# RUN: llvm-mc -filetype=obj -triple mips64 -mcpu=mips64r6 %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s --check-prefix=MIPS64R6

lld $2, 128($sp)
# MIPS64:         d3 a2 00 80  lld    $2, 128($sp)
# MIPS64R6:       7f a2 40 37  lld    $2, 128($sp)

lld $2, -128($sp)
# MIPS64:         d3 a2 ff 80  lld    $2, -128($sp)
# MIPS64R6:       7f a2 c0 37  lld    $2, -128($sp)

lld $2, 256($sp)
# MIPS64:         d3 a2 01 00  lld    $2, 256($sp)

# MIPS64R6:       67 a2 01 00  daddiu $2, $sp, 256
# MIPS64R6-NEXT:  7c 42 00 37  lld    $2, 0($2)

lld $2, -257($sp)
# MIPS64:         d3 a2 fe ff  lld    $2, -257($sp)

# MIPS64R6:       67 a2 fe ff  daddiu $2, $sp, -257
# MIPS64R6-NEXT:  7c 42 00 37  lld    $2, 0($2)

lld $2, 32767($sp)
# MIPS64:         d3 a2 7f ff  lld    $2, 32767($sp)

# MIPS64R6:       67 a2 7f ff  daddiu $2, $sp, 32767
# MIPS64R6-NEXT:  7c 42 00 37  lld    $2, 0($2)

lld $2, 32768($sp)
# MIPS64:         3c 02 00 01  lui    $2, 1
# MIPS64-NEXT:    00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64-NEXT:    d0 42 80 00  lld    $2, -32768($2)

# MIPS64R6:       34 02 80 00  ori    $2, $zero, 32768
# MIPS64R6-NEXT:  00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64R6-NEXT:  7c 42 00 37  lld    $2, 0($2)

lld $2, -32768($sp)
# MIPS64:         d3 a2 80 00  lld    $2, -32768($sp)

# MIPS64R6:       67 a2 80 00  daddiu $2, $sp, -32768
# MIPS64R6-NEXT:  7c 42 00 37  lld    $2, 0($2)

lld $2, -32769($sp)
# MIPS64:         3c 02 ff ff  lui    $2, 65535
# MIPS64-NEXT:    00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64-NEXT:    d0 42 7f ff  lld    $2, 32767($2)

# MIPS64R6:       3c 02 ff ff  aui    $2, $zero, 65535
# MIPS64R6-NEXT:  34 42 7f ff  ori    $2, $2, 32767
# MIPS64R6-NEXT:  00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64R6-NEXT:  7c 42 00 37  lld    $2, 0($2)

lld $2, 2147483648($sp)
# MIPS64:         34 02 80 00  ori    $2, $zero, 32768
# MIPS64-NEXT:    00 02 14 38  dsll   $2, $2, 16
# MIPS64-NEXT:    00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64-NEXT:    d0 42 00 00  lld    $2, 0($2)

# MIPS64R6:       34 02 80 00  ori    $2, $zero, 32768
# MIPS64R6-NEXT:  00 02 14 38  dsll   $2, $2, 16
# MIPS64R6-NEXT:  00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64R6-NEXT:  7c 42 00 37  lld    $2, 0($2)

lld $2, -2147483648($sp)
# MIPS64:         3c 02 80 00  lui    $2, 32768
# MIPS64-NEXT:    00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64-NEXT:    d0 42 00 00  lld    $2, 0($2)

# MIPS64R6:       3c 02 80 00  aui    $2, $zero, 32768
# MIPS64R6-NEXT:  00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64R6-NEXT:  7c 42 00 37  lld    $2, 0($2)

lld $2, 9223372036853775808($sp)
# MIPS64:         3c 02 7f ff  lui    $2, 32767
# MIPS64-NEXT:    34 42 ff ff  ori    $2, $2, 65535
# MIPS64-NEXT:    00 02 14 38  dsll   $2, $2, 16
# MIPS64-NEXT:    34 42 ff f1  ori    $2, $2, 65521
# MIPS64-NEXT:    00 02 14 38  dsll   $2, $2, 16
# MIPS64-NEXT:    00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64-NEXT:    d0 42 bd c0  lld    $2, -16960($2)

# MIPS64R6:       3c 02 7f ff  aui    $2, $zero, 32767
# MIPS64R6-NEXT:  34 42 ff ff  ori    $2, $2, 65535
# MIPS64R6-NEXT:  00 02 14 38  dsll   $2, $2, 16
# MIPS64R6-NEXT:  34 42 ff f0  ori    $2, $2, 65520
# MIPS64R6-NEXT:  00 02 14 38  dsll   $2, $2, 16
# MIPS64R6-NEXT:  34 42 bd c0  ori    $2, $2, 48576
# MIPS64R6-NEXT:  00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64R6-NEXT:  7c 42 00 37  lld    $2, 0($2)

lld $12, symbol
# MIPS64:         3c 0c 00 00  lui    $12, 0
# MIPS64-NEXT:               R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    65 8c 00 00  daddiu $12, $12, 0
# MIPS64-NEXT:               R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    00 0c 64 38  dsll   $12, $12, 16
# MIPS64-NEXT:    65 8c 00 00  daddiu $12, $12, 0
# MIPS64-NEXT:               R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    00 0c 64 38  dsll   $12, $12, 16
# MIPS64-NEXT:    d1 8c 00 00  lld    $12, 0($12)
# MIPS64-NEXT:               R_MIPS_LO16/R_MIPS_NONE/R_MIPS_NONE  symbol

# MIPS64R6:       3c 0c 00 00  aui    $12, $zero, 0
# MIPS64R6-NEXT:             R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64R6-NEXT:  3c 01 00 00  aui    $1, $zero, 0
# MIPS64R6-NEXT:             R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64R6-NEXT:  65 8c 00 00  daddiu $12, $12, 0
# MIPS64R6-NEXT:             R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64R6-NEXT:  64 21 00 00  daddiu $1, $1, 0
# MIPS64R6-NEXT:             R_MIPS_LO16/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64R6-NEXT:  00 0c 60 3c  dsll32 $12, $12, 0
# MIPS64R6-NEXT:  01 81 60 2d  daddu  $12, $12, $1
# MIPS64R6-NEXT:  7d 8c 00 37  lld    $12, 0($12)

lld $12, symbol($3)
# MIPS64:         3c 0c 00 00  lui    $12, 0
# MIPS64-NEXT:               R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    65 8c 00 00  daddiu $12, $12, 0
# MIPS64-NEXT:               R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    00 0c 64 38  dsll   $12, $12, 16
# MIPS64-NEXT:    65 8c 00 00  daddiu $12, $12, 0
# MIPS64-NEXT:               R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    00 0c 64 38  dsll   $12, $12, 16
# MIPS64-NEXT:    01 83 60 2d  daddu  $12, $12, $3
# MIPS64-NEXT:    d1 8c 00 00  lld    $12, 0($12)
# MIPS64-NEXT:               R_MIPS_LO16/R_MIPS_NONE/R_MIPS_NONE  symbol

# MIPS64R6-NEXT:  3c 0c 00 00  aui    $12, $zero, 0
# MIPS64R6-NEXT:             R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64R6-NEXT:  3c 01 00 00  aui    $1, $zero, 0
# MIPS64R6-NEXT:             R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64R6-NEXT:  65 8c 00 00  daddiu $12, $12, 0
# MIPS64R6-NEXT:             R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64R6-NEXT:  64 21 00 00  daddiu $1, $1, 0
# MIPS64R6-NEXT:             R_MIPS_LO16/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64R6-NEXT:  00 0c 60 3c  dsll32 $12, $12, 0
# MIPS64R6-NEXT:  01 81 60 2d  daddu  $12, $12, $1
# MIPS64R6-NEXT:  01 83 60 2d  daddu  $12, $12, $3
# MIPS64R6-NEXT:  7d 8c 00 37  lld    $12, 0($12)

lld $12, symbol+8
# MIPS64:         3c 0c 00 00  lui    $12, 0
# MIPS64-NEXT:               R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64-NEXT:    65 8c 00 00  daddiu $12, $12, 0
# MIPS64-NEXT:               R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64-NEXT:    00 0c 64 38  dsll   $12, $12, 16
# MIPS64-NEXT:    65 8c 00 00  daddiu $12, $12, 0
# MIPS64-NEXT:               R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64-NEXT:    00 0c 64 38  dsll   $12, $12, 16
# MIPS64-NEXT:    d1 8c 00 00  lld    $12, 0($12)
# MIPS64-NEXT:               R_MIPS_LO16/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8

# MIPS64R6-NEXT:  3c 0c 00 00  aui    $12, $zero, 0
# MIPS64R6-NEXT:             R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64R6-NEXT:  3c 01 00 00  aui    $1, $zero, 0
# MIPS64R6-NEXT:             R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64R6-NEXT:  65 8c 00 00  daddiu $12, $12, 0
# MIPS64R6-NEXT:             R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64R6-NEXT:  64 21 00 00  daddiu $1, $1, 0
# MIPS64R6-NEXT:             R_MIPS_LO16/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64R6-NEXT:  00 0c 60 3c  dsll32 $12, $12, 0
# MIPS64R6-NEXT:  01 81 60 2d  daddu  $12, $12, $1
# MIPS64R6-NEXT:  7d 8c 00 37  lld    $12, 0($12)

.option pic2

lld $12, symbol
# MIPS64:         df 8c 00 00  ld     $12, 0($gp)
# MIPS64-NEXT:               R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE symbol
# MIPS64-NEXT:    d1 8c 00 00  lld    $12, 0($12)

# MIPS64R6:       df 8c 00 00  ld     $12, 0($gp)
# MIPS64R6-NEXT:             R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE symbol
# MIPS64R6-NEXT:  7d 8c 00 37  lld    $12, 0($12)

lld $12, symbol+8
# MIPS64:         df 8c 00 00  ld     $12, 0($gp)
# MIPS64-NEXT:               R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE symbol
# MIPS64-NEXT:    d1 8c 00 08  lld    $12, 8($12)

# MIPS64R6:       df 8c 00 00  ld     $12, 0($gp)
# MIPS64R6-NEXT:             R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE symbol
# MIPS64R6-NEXT:  65 8c 00 08  daddiu $12, $12, 8
# MIPS64R6-NEXT:  7d 8c 00 37  lld    $12, 0($12)
