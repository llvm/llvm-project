# RUN: llvm-mc -filetype=obj -triple mips -mcpu=mips2 %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s --check-prefix=MIPS32
# RUN: llvm-mc -filetype=obj -triple mips -mcpu=mips32 %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s --check-prefix=MIPS32
# RUN: llvm-mc -filetype=obj -triple mips -mcpu=mips32r2 %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s --check-prefix=MIPS32
# RUN: llvm-mc -filetype=obj -triple mipsn32 -mcpu=mips3 %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s --check-prefix=MIPSN32
# RUN: llvm-mc -filetype=obj -triple mipsn32 -mcpu=mips64r6 %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s --check-prefix=MIPSN32R6
# RUN: llvm-mc -filetype=obj -triple mips64 -mcpu=mips64 %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s --check-prefix=MIPS64
# RUN: llvm-mc -filetype=obj -triple mips64 -mcpu=mips64r2 %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s --check-prefix=MIPS64
# RUN: llvm-mc -filetype=obj -triple mips -mcpu=mips32r6 %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s --check-prefix=MIPS32R6
# RUN: llvm-mc -filetype=obj -triple mips64 -mcpu=mips64r6 %s -o - \
# RUN:   | llvm-objdump --no-print-imm-hex -d -r - | FileCheck %s --check-prefix=MIPS64R6

ll $2, 128($sp)
# MIPS32:         c3 a2 00 80  ll     $2, 128($sp)
# MIPS32R6:       7f a2 40 36  ll     $2, 128($sp)
# MIPSN32:        c3 a2 00 80  ll     $2, 128($sp)
# MIPSN32R6:      7f a2 40 36  ll     $2, 128($sp)
# MIPS64:         c3 a2 00 80  ll     $2, 128($sp)
# MIPS64R6:       7f a2 40 36  ll     $2, 128($sp)

ll $2, -128($sp)
# MIPS32:         c3 a2 ff 80  ll     $2, -128($sp)
# MIPS32R6:       7f a2 c0 36  ll     $2, -128($sp)
# MIPSN32:        c3 a2 ff 80  ll     $2, -128($sp)
# MIPSN32R6:      7f a2 c0 36  ll     $2, -128($sp)
# MIPS64:         c3 a2 ff 80  ll     $2, -128($sp)
# MIPS64R6:       7f a2 c0 36  ll     $2, -128($sp)

ll $2, 256($sp)
# MIPS32:         c3 a2 01 00  ll     $2, 256($sp)

# MIPS32R6:       27 a2 01 00  addiu  $2, $sp, 256
# MIPS32R6-NEXT:  7c 42 00 36  ll     $2, 0($2)

# MIPSN32:        c3 a2 01 00  ll     $2, 256($sp)

# MIPSN32R6:      27 a2 01 00  addiu  $2, $sp, 256
# MIPSN32R6-NEXT: 7c 42 00 36  ll     $2, 0($2)

# MIPS64:         c3 a2 01 00  ll     $2, 256($sp)

# MIPS64R6:       67 a2 01 00  daddiu $2, $sp, 256
# MIPS64R6-NEXT:  7c 42 00 36  ll     $2, 0($2)

ll $2, -257($sp)
# MIPS32:         c3 a2 fe ff  ll     $2, -257($sp)

# MIPS32R6:       27 a2 fe ff  addiu  $2, $sp, -257
# MIPS32R6-NEXT:  7c 42 00 36  ll     $2, 0($2)

# MIPSN32:        c3 a2 fe ff  ll     $2, -257($sp)

# MIPSN32R6:      27 a2 fe ff  addiu  $2, $sp, -257
# MIPSN32R6-NEXT: 7c 42 00 36  ll     $2, 0($2)

# MIPS64:         c3 a2 fe ff  ll     $2, -257($sp)

# MIPS64R6:       67 a2 fe ff  daddiu $2, $sp, -257
# MIPS64R6-NEXT:  7c 42 00 36  ll     $2, 0($2)

ll $2, 32767($sp)
# MIPS32:         c3 a2 7f ff  ll     $2, 32767($sp)

# MIPS32R6:       27 a2 7f ff  addiu  $2, $sp, 32767
# MIPS32R6-NEXT:  7c 42 00 36  ll     $2, 0($2)

# MIPSN32:        c3 a2 7f ff  ll     $2, 32767($sp)

# MIPSN32R6:      27 a2 7f ff  addiu  $2, $sp, 32767
# MIPSN32R6-NEXT: 7c 42 00 36  ll     $2, 0($2)

# MIPS64:         c3 a2 7f ff  ll     $2, 32767($sp)

# MIPS64R6:       67 a2 7f ff  daddiu $2, $sp, 32767
# MIPS64R6-NEXT:  7c 42 00 36  ll     $2, 0($2)

ll $2, 32768($sp)
# MIPS32:         3c 02 00 01  lui    $2, 1
# MIPS32-NEXT:    00 5d 10 21  addu   $2, $2, $sp
# MIPS32-NEXT:    c0 42 80 00  ll     $2, -32768($2)

# MIPS32R6:       34 02 80 00  ori    $2, $zero, 32768
# MIPS32R6-NEXT:  00 5d 10 21  addu   $2, $2, $sp
# MIPS32R6-NEXT:  7c 42 00 36  ll     $2, 0($2)

# MIPSN32:        3c 02 00 01  lui    $2, 1
# MIPSN32-NEXT:   00 5d 10 21  addu   $2, $2, $sp
# MIPSN32-NEXT:   c0 42 80 00  ll     $2, -32768($2)

# MIPSN32R6:      34 02 80 00  ori    $2, $zero, 32768
# MIPSN32R6-NEXT: 00 5d 10 21  addu   $2, $2, $sp
# MIPSN32R6-NEXT: 7c 42 00 36  ll     $2, 0($2)

# MIPS64:         3c 02 00 01  lui    $2, 1
# MIPS64-NEXT:    00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64-NEXT:    c0 42 80 00  ll     $2, -32768($2)

# MIPS64R6:       34 02 80 00  ori    $2, $zero, 32768
# MIPS64R6-NEXT:  00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64R6-NEXT:  7c 42 00 36  ll     $2, 0($2)

ll $2, -32768($sp)
# MIPS32:         c3 a2 80 00  ll     $2, -32768($sp)

# MIPS32R6:       27 a2 80 00  addiu  $2, $sp, -32768
# MIPS32R6-NEXT:  7c 42 00 36  ll     $2, 0($2)

# MIPSN32:        c3 a2 80 00  ll     $2, -32768($sp)

# MIPSN32R6:      27 a2 80 00  addiu  $2, $sp, -32768
# MIPSN32R6-NEXT: 7c 42 00 36  ll     $2, 0($2)

# MIPS64:         c3 a2 80 00  ll     $2, -32768($sp)

# MIPS64R6:       67 a2 80 00  daddiu $2, $sp, -32768
# MIPS64R6-NEXT:  7c 42 00 36  ll     $2, 0($2)

ll $2, -32769($sp)
# MIPS32:         3c 02 ff ff  lui    $2, 65535
# MIPS32-NEXT:    00 5d 10 21  addu   $2, $2, $sp
# MIPS32-NEXT:    c0 42 7f ff  ll     $2, 32767($2)

# MIPS32R6:       3c 02 ff ff  aui    $2, $zero, 65535
# MIPS32R6-NEXT:  34 42 7f ff  ori    $2, $2, 32767
# MIPS32R6-NEXT:  00 5d 10 21  addu   $2, $2, $sp
# MIPS32R6-NEXT:  7c 42 00 36  ll     $2, 0($2)

# MIPSN32:        3c 02 ff ff  lui    $2, 65535
# MIPSN32-NEXT:   00 5d 10 21  addu   $2, $2, $sp
# MIPSN32-NEXT:   c0 42 7f ff  ll     $2, 32767($2)

# MIPSN32R6:      3c 02 ff ff  aui    $2, $zero, 65535
# MIPSN32R6-NEXT: 34 42 7f ff  ori    $2, $2, 32767
# MIPSN32R6-NEXT: 00 5d 10 21  addu   $2, $2, $sp
# MIPSN32R6-NEXT: 7c 42 00 36  ll     $2, 0($2)

# MIPS64:         3c 02 ff ff  lui    $2, 65535
# MIPS64-NEXT:    00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64-NEXT:    c0 42 7f ff  ll     $2, 32767($2)

# MIPS64R6:       3c 02 ff ff  aui    $2, $zero, 65535
# MIPS64R6-NEXT:  34 42 7f ff  ori    $2, $2, 32767
# MIPS64R6-NEXT:  00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64R6-NEXT:  7c 42 00 36  ll     $2, 0($2)

ll $2, 655987($sp)
# MIPS32:         3c 02 00 0a  lui    $2, 10
# MIPS32-NEXT:    00 5d 10 21  addu   $2, $2, $sp
# MIPS32-NEXT:    c0 42 02 73  ll     $2, 627($2)

# MIPS32R6:       3c 02 00 0a  aui    $2, $zero, 10
# MIPS32R6-NEXT:  34 42 02 73  ori    $2, $2, 627
# MIPS32R6-NEXT:  00 5d 10 21  addu   $2, $2, $sp
# MIPS32R6-NEXT:  7c 42 00 36  ll     $2, 0($2)

# MIPSN32:        3c 02 00 0a  lui    $2, 10
# MIPSN32-NEXT:   00 5d 10 21  addu   $2, $2, $sp
# MIPSN32-NEXT:   c0 42 02 73  ll     $2, 627($2)

# MIPSN32R6:      3c 02 00 0a  aui    $2, $zero, 10
# MIPSN32R6-NEXT: 34 42 02 73  ori    $2, $2, 627
# MIPSN32R6-NEXT: 00 5d 10 21  addu   $2, $2, $sp
# MIPSN32R6-NEXT: 7c 42 00 36  ll     $2, 0($2)

# MIPS64:         3c 02 00 0a  lui    $2, 10
# MIPS64-NEXT:    00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64-NEXT:    c0 42 02 73  ll     $2, 627($2)

# MIPS64R6:       3c 02 00 0a  aui    $2, $zero, 10
# MIPS64R6-NEXT:  34 42 02 73  ori    $2, $2, 627
# MIPS64R6-NEXT:  00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64R6-NEXT:  7c 42 00 36  ll     $2, 0($2)

ll $2, -655987($sp)
# MIPS32:         3c 02 ff f6  lui    $2, 65526
# MIPS32-NEXT:    00 5d 10 21  addu   $2, $2, $sp
# MIPS32-NEXT:    c0 42 fd 8d  ll     $2, -627($2)

# MIPS32R6:       3c 02 ff f5  aui    $2, $zero, 65525
# MIPS32R6-NEXT:  34 42 fd 8d  ori    $2, $2, 64909
# MIPS32R6-NEXT:  00 5d 10 21  addu   $2, $2, $sp
# MIPS32R6-NEXT:  7c 42 00 36  ll     $2, 0($2)

# MIPSN32:        3c 02 ff f6  lui    $2, 65526
# MIPSN32-NEXT:   00 5d 10 21  addu   $2, $2, $sp
# MIPSN32-NEXT:   c0 42 fd 8d  ll     $2, -627($2)

# MIPSN32R6:      3c 02 ff f5  aui    $2, $zero, 65525
# MIPSN32R6-NEXT: 34 42 fd 8d  ori    $2, $2, 64909
# MIPSN32R6-NEXT: 00 5d 10 21  addu   $2, $2, $sp
# MIPSN32R6-NEXT: 7c 42 00 36  ll     $2, 0($2)

# MIPS64:         3c 02 ff f6  lui    $2, 65526
# MIPS64-NEXT:    00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64-NEXT:    c0 42 fd 8d  ll     $2, -627($2)

# MIPS64R6:       3c 02 ff f5  aui    $2, $zero, 65525
# MIPS64R6-NEXT:  34 42 fd 8d  ori    $2, $2, 64909
# MIPS64R6-NEXT:  00 5d 10 2d  daddu  $2, $2, $sp
# MIPS64R6-NEXT:  7c 42 00 36  ll     $2, 0($2)

ll $12, symbol
# MIPS32:         3c 0c 00 00  lui    $12, 0
# MIPS32-NEXT:               R_MIPS_HI16  symbol
# MIPS32-NEXT:    c1 8c 00 00  ll     $12, 0($12)
# MIPS32-NEXT:               R_MIPS_LO16  symbol

# MIPS32R6:       3c 0c 00 00  aui    $12, $zero, 0
# MIPS32R6-NEXT:             R_MIPS_HI16 symbol
# MIPS32R6-NEXT:  25 8c 00 00  addiu  $12, $12, 0
# MIPS32R6-NEXT:             R_MIPS_LO16 symbol
# MIPS32R6-NEXT:  7d 8c 00 36  ll     $12, 0($12)

# MIPSN32:        3c 0c 00 00  lui    $12, 0
# MIPSN32-NEXT:              R_MIPS_HI16  symbol
# MIPSN32-NEXT:   c1 8c 00 00  ll     $12, 0($12)
# MIPSN32-NEXT:              R_MIPS_LO16  symbol

# MIPSN32R6:      3c 0c 00 00  aui    $12, $zero, 0
# MIPSN32R6-NEXT:            R_MIPS_HI16 symbol
# MIPSN32R6-NEXT: 25 8c 00 00  addiu  $12, $12, 0
# MIPSN32R6-NEXT:            R_MIPS_LO16 symbol
# MIPSN32R6-NEXT: 7d 8c 00 36  ll     $12, 0($12)

# MIPS64:         3c 0c 00 00  lui    $12, 0
# MIPS64-NEXT:               R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    65 8c 00 00  daddiu $12, $12, 0
# MIPS64-NEXT:               R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    00 0c 64 38  dsll   $12, $12, 16
# MIPS64-NEXT:    65 8c 00 00  daddiu $12, $12, 0
# MIPS64-NEXT:               R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    00 0c 64 38  dsll   $12, $12, 16
# MIPS64-NEXT:    c1 8c 00 00  ll     $12, 0($12)
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
# MIPS64R6-NEXT:  7d 8c 00 36  ll     $12, 0($12)

ll $12, symbol($3)
# MIPS32:         3c 0c 00 00  lui    $12, 0
# MIPS32-NEXT:               R_MIPS_HI16  symbol
# MIPS32-NEXT:    01 83 60 21  addu   $12, $12, $3
# MIPS32-NEXT:    c1 8c 00 00  ll     $12, 0($12)
# MIPS32-NEXT:               R_MIPS_LO16  symbol

# MIPS32R6:       3c 0c 00 00  aui    $12, $zero, 0
# MIPS32R6-NEXT:             R_MIPS_HI16 symbol
# MIPS32R6-NEXT:  25 8c 00 00  addiu  $12, $12, 0
# MIPS32R6-NEXT:             R_MIPS_LO16 symbol
# MIPS32R6-NEXT:  01 83 60 21  addu   $12, $12, $3
# MIPS32R6-NEXT:  7d 8c 00 36  ll     $12, 0($12)

# MIPSN32:        3c 0c 00 00  lui    $12, 0
# MIPSN32-NEXT:              R_MIPS_HI16  symbol
# MIPSN32-NEXT:   01 83 60 21  addu   $12, $12, $3
# MIPSN32-NEXT:   c1 8c 00 00  ll     $12, 0($12)
# MIPSN32-NEXT:              R_MIPS_LO16  symbol

# MIPSN32R6:      3c 0c 00 00  aui    $12, $zero, 0
# MIPSN32R6-NEXT:            R_MIPS_HI16 symbol
# MIPSN32R6-NEXT: 25 8c 00 00  addiu  $12, $12, 0
# MIPSN32R6-NEXT:            R_MIPS_LO16 symbol
# MIPSN32R6-NEXT: 01 83 60 21  addu   $12, $12, $3
# MIPSN32R6-NEXT: 7d 8c 00 36  ll     $12, 0($12)

# MIPS64:         3c 0c 00 00  lui    $12, 0
# MIPS64-NEXT:               R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    65 8c 00 00  daddiu $12, $12, 0
# MIPS64-NEXT:               R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    00 0c 64 38  dsll   $12, $12, 16
# MIPS64-NEXT:    65 8c 00 00  daddiu $12, $12, 0
# MIPS64-NEXT:               R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    00 0c 64 38  dsll   $12, $12, 16
# MIPS64-NEXT:    01 83 60 2d  daddu  $12, $12, $3
# MIPS64-NEXT:    c1 8c 00 00  ll     $12, 0($12)
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
# MIPS64R6-NEXT:  01 83 60 2d  daddu  $12, $12, $3
# MIPS64R6-NEXT:  7d 8c 00 36  ll     $12, 0($12)

ll $12, symbol+8
# MIPS32:         3c 0c 00 00  lui    $12, 0
# MIPS32-NEXT:               R_MIPS_HI16  symbol
# MIPS32-NEXT:    c1 8c 00 08  ll     $12, 8($12)
# MIPS32-NEXT:               R_MIPS_LO16  symbol

# MIPS32R6:       3c 0c 00 00  aui    $12, $zero, 0
# MIPS32R6-NEXT:             R_MIPS_HI16 symbol
# MIPS32R6-NEXT:  25 8c 00 08  addiu  $12, $12, 8
# MIPS32R6-NEXT:             R_MIPS_LO16 symbol
# MIPS32R6-NEXT:  7d 8c 00 36  ll     $12, 0($12)

# MIPSN32:        3c 0c 00 00  lui    $12, 0
# MIPSN32-NEXT:              R_MIPS_HI16  symbol+0x8
# MIPSN32-NEXT:   c1 8c 00 00  ll     $12, 0($12)
# MIPSN32-NEXT:              R_MIPS_LO16  symbol+0x8

# MIPSN32R6:      3c 0c 00 00  aui    $12, $zero, 0
# MIPSN32R6-NEXT:            R_MIPS_HI16 symbol+0x8
# MIPSN32R6-NEXT: 25 8c 00 00  addiu  $12, $12, 0
# MIPSN32R6-NEXT:            R_MIPS_LO16 symbol+0x8
# MIPSN32R6-NEXT: 7d 8c 00 36  ll     $12, 0($12)

# MIPS64:         3c 0c 00 00  lui    $12, 0
# MIPS64-NEXT:               R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64-NEXT:    65 8c 00 00  daddiu $12, $12, 0
# MIPS64-NEXT:               R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64-NEXT:    00 0c 64 38  dsll   $12, $12, 16
# MIPS64-NEXT:    65 8c 00 00  daddiu $12, $12, 0
# MIPS64-NEXT:               R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64-NEXT:    00 0c 64 38  dsll   $12, $12, 16
# MIPS64-NEXT:    c1 8c 00 00  ll     $12, 0($12)
# MIPS64-NEXT:               R_MIPS_LO16/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8

# MIPS64R6:       3c 0c 00 00  aui    $12, $zero, 0
# MIPS64R6-NEXT:             R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64R6-NEXT:  3c 01 00 00  aui    $1, $zero, 0
# MIPS64R6-NEXT:             R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64R6-NEXT:  65 8c 00 00  daddiu $12, $12, 0
# MIPS64R6-NEXT:             R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64R6-NEXT:  64 21 00 00  daddiu $1, $1, 0
# MIPS64R6-NEXT:             R_MIPS_LO16/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64R6-NEXT:  00 0c 60 3c  dsll32 $12, $12, 0
# MIPS64R6-NEXT:  01 81 60 2d  daddu  $12, $12, $1
# MIPS64R6-NEXT:  7d 8c 00 36  ll     $12, 0($12)

.option pic2

ll $12, symbol
# MIPS32:         8f 8c 00 00  lw     $12, 0($gp)
# MIPS32-NEXT:               R_MIPS_GOT16 symbol
# MIPS32-NEXT:    c1 8c 00 00  ll     $12, 0($12)

# MIPS32R6:       8f 8c 00 00  lw     $12, 0($gp)
# MIPS32R6-NEXT:             R_MIPS_GOT16 symbol
# MIPS32R6-NEXT:  7d 8c 00 36  ll     $12, 0($12)

# MIPSN32:        8f 8c 00 00  lw     $12, 0($gp)
# MIPSN32-NEXT:              R_MIPS_GOT_DISP symbol
# MIPSN32-NEXT:   c1 8c 00 00  ll     $12, 0($12)

# MIPSN32R6:      8f 8c 00 00  lw     $12, 0($gp)
# MIPSN32R6-NEXT:            R_MIPS_GOT_DISP symbol
# MIPSN32R6-NEXT: 7d 8c 00 36  ll     $12, 0($12)

# MIPS64:         df 8c 00 00  ld     $12, 0($gp)
# MIPS64-NEXT:               R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE symbol
# MIPS64-NEXT:    c1 8c 00 00  ll     $12, 0($12)

# MIPS64R6:       df 8c 00 00  ld     $12, 0($gp)
# MIPS64R6-NEXT:             R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE symbol
# MIPS64R6-NEXT:  7d 8c 00 36  ll     $12, 0($12)

ll $12, symbol+8
# MIPS32:         8f 8c 00 00  lw     $12, 0($gp)
# MIPS32-NEXT:               R_MIPS_GOT16 symbol
# MIPS32-NEXT:    c1 8c 00 08  ll     $12, 8($12)

# MIPS32R6:       8f 8c 00 00  lw     $12, 0($gp)
# MIPS32R6-NEXT:             R_MIPS_GOT16 symbol
# MIPS32R6-NEXT:  25 8c 00 08  addiu  $12, $12, 8
# MIPS32R6-NEXT:  7d 8c 00 36  ll     $12, 0($12)

# MIPSN32:        8f 8c 00 00  lw     $12, 0($gp)
# MIPSN32-NEXT:              R_MIPS_GOT_DISP symbol
# MIPSN32-NEXT:   c1 8c 00 08  ll     $12, 8($12)

# MIPSN32R6:      8f 8c 00 00  lw     $12, 0($gp)
# MIPSN32R6-NEXT:            R_MIPS_GOT_DISP symbol
# MIPSN32R6-NEXT: 25 8c 00 08  addiu  $12, $12, 8
# MIPSN32R6-NEXT: 7d 8c 00 36  ll     $12, 0($12)

# MIPS64:         df 8c 00 00  ld     $12, 0($gp)
# MIPS64-NEXT:               R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE symbol
# MIPS64-NEXT:    c1 8c 00 08  ll     $12, 8($12)

# MIPS64R6:       df 8c 00 00  ld     $12, 0($gp)
# MIPS64R6-NEXT:             R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE symbol
# MIPS64R6-NEXT:  65 8c 00 08  daddiu $12, $12, 8
# MIPS64R6-NEXT:  7d 8c 00 36  ll     $12, 0($12)
