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

sc $2, 128($sp)
# MIPS32:         e3 a2 00 80  sc     $2, 128($sp)
# MIPS32R6:       7f a2 40 26  sc     $2, 128($sp)
# MIPSN32:        e3 a2 00 80  sc     $2, 128($sp)
# MIPSN32R6:      7f a2 40 26  sc     $2, 128($sp)
# MIPS64:         e3 a2 00 80  sc     $2, 128($sp)
# MIPS64R6:       7f a2 40 26  sc     $2, 128($sp)

sc $2, -128($sp)
# MIPS32:         e3 a2 ff 80  sc     $2, -128($sp)
# MIPS32R6:       7f a2 c0 26  sc     $2, -128($sp)
# MIPSN32:        e3 a2 ff 80  sc     $2, -128($sp)
# MIPSN32R6:      7f a2 c0 26  sc     $2, -128($sp)
# MIPS64:         e3 a2 ff 80  sc     $2, -128($sp)
# MIPS64R6:       7f a2 c0 26  sc     $2, -128($sp)

sc $2, 256($sp)
# MIPS32:         e3 a2 01 00  sc     $2, 256($sp)

# MIPS32R6:       27 a1 01 00  addiu  $1, $sp, 256
# MIPS32R6-NEXT:  7c 22 00 26  sc     $2, 0($1)

# MIPSN32:        e3 a2 01 00  sc     $2, 256($sp)

# MIPSN32R6:      27 a1 01 00  addiu  $1, $sp, 256
# MIPSN32R6-NEXT: 7c 22 00 26  sc     $2, 0($1)

# MIPS64:         e3 a2 01 00  sc     $2, 256($sp)

# MIPS64R6:       67 a1 01 00  daddiu $1, $sp, 256
# MIPS64R6-NEXT:  7c 22 00 26  sc     $2, 0($1)

sc $2, -257($sp)
# MIPS32:         e3 a2 fe ff  sc     $2, -257($sp)

# MIPS32R6:       27 a1 fe ff  addiu  $1, $sp, -257
# MIPS32R6-NEXT:  7c 22 00 26  sc     $2, 0($1)

# MIPSN32:        e3 a2 fe ff  sc     $2, -257($sp)

# MIPSN32R6:      27 a1 fe ff  addiu  $1, $sp, -257
# MIPSN32R6-NEXT: 7c 22 00 26  sc     $2, 0($1)

# MIPS64:         e3 a2 fe ff  sc     $2, -257($sp)

# MIPS64R6:       67 a1 fe ff  daddiu $1, $sp, -257
# MIPS64R6-NEXT:  7c 22 00 26  sc     $2, 0($1)

sc $2, 32767($sp)
# MIPS32:         e3 a2 7f ff  sc     $2, 32767($sp)

# MIPS32R6:       27 a1 7f ff  addiu  $1, $sp, 32767
# MIPS32R6-NEXT:  7c 22 00 26  sc     $2, 0($1)

# MIPSN32:        e3 a2 7f ff  sc     $2, 32767($sp)

# MIPSN32R6:      27 a1 7f ff  addiu  $1, $sp, 32767
# MIPSN32R6-NEXT: 7c 22 00 26  sc     $2, 0($1)

# MIPS64:         e3 a2 7f ff  sc     $2, 32767($sp)

# MIPS64R6:       67 a1 7f ff  daddiu $1, $sp, 32767
# MIPS64R6-NEXT:  7c 22 00 26  sc     $2, 0($1)

sc $2, 32768($sp)
# MIPS32:         3c 01 00 01  lui    $1, 1
# MIPS32-NEXT:    00 3d 08 21  addu   $1, $1, $sp
# MIPS32-NEXT:    e0 22 80 00  sc     $2, -32768($1)

# MIPS32R6:       34 01 80 00  ori    $1, $zero, 32768
# MIPS32R6-NEXT:  00 3d 08 21  addu   $1, $1, $sp
# MIPS32R6-NEXT:  7c 22 00 26  sc     $2, 0($1)

# MIPSN32:        3c 01 00 01  lui    $1, 1
# MIPSN32-NEXT:   00 3d 08 21  addu   $1, $1, $sp
# MIPSN32-NEXT:   e0 22 80 00  sc     $2, -32768($1)

# MIPSN32R6:      34 01 80 00  ori    $1, $zero, 32768
# MIPSN32R6-NEXT: 00 3d 08 21  addu   $1, $1, $sp
# MIPSN32R6-NEXT: 7c 22 00 26  sc     $2, 0($1)

# MIPS64:         3c 01 00 01  lui    $1, 1
# MIPS64-NEXT:    00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64-NEXT:    e0 22 80 00  sc     $2, -32768($1)

# MIPS64R6:       34 01 80 00  ori    $1, $zero, 32768
# MIPS64R6-NEXT:  00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64R6-NEXT:  7c 22 00 26  sc     $2, 0($1)

sc $2, -32768($sp)
# MIPS32:         e3 a2 80 00  sc     $2, -32768($sp)

# MIPS32R6:       27 a1 80 00  addiu  $1, $sp, -32768
# MIPS32R6-NEXT:  7c 22 00 26  sc     $2, 0($1)

# MIPSN32:        e3 a2 80 00  sc     $2, -32768($sp)

# MIPSN32R6:      27 a1 80 00  addiu  $1, $sp, -32768
# MIPSN32R6-NEXT: 7c 22 00 26  sc     $2, 0($1)

# MIPS64:         e3 a2 80 00  sc     $2, -32768($sp)

# MIPS64R6:       67 a1 80 00  daddiu $1, $sp, -32768
# MIPS64R6-NEXT:  7c 22 00 26  sc     $2, 0($1)

sc $2, -32769($sp)
# MIPS32:         3c 01 ff ff  lui    $1, 65535
# MIPS32-NEXT:    00 3d 08 21  addu   $1, $1, $sp
# MIPS32-NEXT:    e0 22 7f ff  sc     $2, 32767($1)

# MIPS32R6:       3c 01 ff ff  aui    $1, $zero, 65535
# MIPS32R6-NEXT:  34 21 7f ff  ori    $1, $1, 32767
# MIPS32R6-NEXT:  00 3d 08 21  addu   $1, $1, $sp
# MIPS32R6-NEXT:  7c 22 00 26  sc     $2, 0($1)

# MIPSN32:        3c 01 ff ff  lui    $1, 65535
# MIPSN32-NEXT:   00 3d 08 21  addu   $1, $1, $sp
# MIPSN32-NEXT:   e0 22 7f ff  sc     $2, 32767($1)

# MIPSN32R6:      3c 01 ff ff  aui    $1, $zero, 65535
# MIPSN32R6-NEXT: 34 21 7f ff  ori    $1, $1, 32767
# MIPSN32R6-NEXT: 00 3d 08 21  addu   $1, $1, $sp
# MIPSN32R6-NEXT: 7c 22 00 26  sc     $2, 0($1)

# MIPS64:         3c 01 ff ff  lui    $1, 65535
# MIPS64-NEXT:    00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64-NEXT:    e0 22 7f ff  sc     $2, 32767($1)

# MIPS64R6:       3c 01 ff ff  aui    $1, $zero, 65535
# MIPS64R6-NEXT:  34 21 7f ff  ori    $1, $1, 32767
# MIPS64R6-NEXT:  00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64R6-NEXT:  7c 22 00 26  sc     $2, 0($1)

sc $2, 655987($sp)
# MIPS32:         3c 01 00 0a  lui    $1, 10
# MIPS32-NEXT:    00 3d 08 21  addu   $1, $1, $sp
# MIPS32-NEXT:    e0 22 02 73  sc     $2, 627($1)

# MIPS32R6:       3c 01 00 0a  aui    $1, $zero, 10
# MIPS32R6-NEXT:  34 21 02 73  ori    $1, $1, 627
# MIPS32R6-NEXT:  00 3d 08 21  addu   $1, $1, $sp
# MIPS32R6-NEXT:  7c 22 00 26  sc     $2, 0($1)

# MIPSN32:        3c 01 00 0a  lui    $1, 10
# MIPSN32-NEXT:   00 3d 08 21  addu   $1, $1, $sp
# MIPSN32-NEXT:   e0 22 02 73  sc     $2, 627($1)

# MIPSN32R6:      3c 01 00 0a  aui    $1, $zero, 10
# MIPSN32R6-NEXT: 34 21 02 73  ori    $1, $1, 627
# MIPSN32R6-NEXT: 00 3d 08 21  addu   $1, $1, $sp
# MIPSN32R6-NEXT: 7c 22 00 26  sc     $2, 0($1)

# MIPS64:         3c 01 00 0a  lui    $1, 10
# MIPS64-NEXT:    00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64-NEXT:    e0 22 02 73  sc     $2, 627($1)

# MIPS64R6:       3c 01 00 0a  aui    $1, $zero, 10
# MIPS64R6-NEXT:  34 21 02 73  ori    $1, $1, 627
# MIPS64R6-NEXT:  00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64R6-NEXT:  7c 22 00 26  sc     $2, 0($1)

sc $2, -655987($sp)
# MIPS32:         3c 01 ff f6  lui    $1, 65526
# MIPS32-NEXT:    00 3d 08 21  addu   $1, $1, $sp
# MIPS32-NEXT:    e0 22 fd 8d  sc     $2, -627($1)

# MIPS32R6:       3c 01 ff f5  aui    $1, $zero, 65525
# MIPS32R6-NEXT:  34 21 fd 8d  ori    $1, $1, 64909
# MIPS32R6-NEXT:  00 3d 08 21  addu   $1, $1, $sp
# MIPS32R6-NEXT:  7c 22 00 26  sc     $2, 0($1)

# MIPSN32:        3c 01 ff f6  lui    $1, 65526
# MIPSN32-NEXT:   00 3d 08 21  addu   $1, $1, $sp
# MIPSN32-NEXT:   e0 22 fd 8d  sc     $2, -627($1)

# MIPSN32R6:      3c 01 ff f5  aui    $1, $zero, 65525
# MIPSN32R6-NEXT: 34 21 fd 8d  ori    $1, $1, 64909
# MIPSN32R6-NEXT: 00 3d 08 21  addu   $1, $1, $sp
# MIPSN32R6-NEXT: 7c 22 00 26  sc     $2, 0($1)

# MIPS64:         3c 01 ff f6  lui    $1, 65526
# MIPS64-NEXT:    00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64-NEXT:    e0 22 fd 8d  sc     $2, -627($1)

# MIPS64R6:       3c 01 ff f5  aui    $1, $zero, 65525
# MIPS64R6-NEXT:  34 21 fd 8d  ori    $1, $1, 64909
# MIPS64R6-NEXT:  00 3d 08 2d  daddu  $1, $1, $sp
# MIPS64R6-NEXT:  7c 22 00 26  sc     $2, 0($1)

sc $12, symbol
# MIPS32:         3c 01 00 00  lui    $1, 0
# MIPS32-NEXT:               R_MIPS_HI16  symbol
# MIPS32-NEXT:    e0 2c 00 00  sc     $12, 0($1)
# MIPS32-NEXT:               R_MIPS_LO16  symbol

# MIPS32R6:       3c 01 00 00  aui    $1, $zero, 0
# MIPS32R6-NEXT:             R_MIPS_HI16 symbol
# MIPS32R6-NEXT:  24 21 00 00  addiu  $1, $1, 0
# MIPS32R6-NEXT:             R_MIPS_LO16 symbol
# MIPS32R6-NEXT:  7c 2c 00 26  sc     $12, 0($1)

# MIPSN32:        3c 01 00 00  lui    $1, 0
# MIPSN32-NEXT:              R_MIPS_HI16  symbol
# MIPSN32-NEXT:   e0 2c 00 00  sc     $12, 0($1)
# MIPSN32-NEXT:              R_MIPS_LO16  symbol

# MIPSN32R6:      3c 01 00 00  aui    $1, $zero, 0
# MIPSN32R6-NEXT:            R_MIPS_HI16 symbol
# MIPSN32R6-NEXT: 24 21 00 00  addiu  $1, $1, 0
# MIPSN32R6-NEXT:            R_MIPS_LO16 symbol
# MIPSN32R6-NEXT: 7c 2c 00 26  sc     $12, 0($1)

# MIPS64:         3c 01 00 00  lui    $1, 0
# MIPS64-NEXT:               R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    64 21 00 00  daddiu $1, $1, 0
# MIPS64-NEXT:               R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    00 01 0c 38  dsll   $1, $1, 16
# MIPS64-NEXT:    64 21 00 00  daddiu $1, $1, 0
# MIPS64-NEXT:               R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    00 01 0c 38  dsll   $1, $1, 16
# MIPS64-NEXT:    e0 2c 00 00  sc     $12, 0($1)
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
# MIPS64R6-NEXT:  7c 2c 00 26  sc     $12, 0($1)

sc $12, symbol($3)
# MIPS32:         3c 01 00 00  lui    $1, 0
# MIPS32-NEXT:               R_MIPS_HI16  symbol
# MIPS32-NEXT:    00 23 08 21  addu   $1, $1, $3
# MIPS32-NEXT:    e0 2c 00 00  sc     $12, 0($1)
# MIPS32-NEXT:               R_MIPS_LO16  symbol

# MIPS32R6:       3c 01 00 00  aui    $1, $zero, 0
# MIPS32R6-NEXT:             R_MIPS_HI16 symbol
# MIPS32R6-NEXT:  24 21 00 00  addiu  $1, $1, 0
# MIPS32R6-NEXT:             R_MIPS_LO16 symbol
# MIPS32R6-NEXT:  00 23 08 21  addu   $1, $1, $3
# MIPS32R6-NEXT:  7c 2c 00 26  sc     $12, 0($1)

# MIPSN32:        3c 01 00 00  lui    $1, 0
# MIPSN32-NEXT:              R_MIPS_HI16  symbol
# MIPSN32-NEXT:   00 23 08 21  addu   $1, $1, $3
# MIPSN32-NEXT:   e0 2c 00 00  sc     $12, 0($1)
# MIPSN32-NEXT:              R_MIPS_LO16  symbol

# MIPSN32R6:      3c 01 00 00  aui    $1, $zero, 0
# MIPSN32R6-NEXT:            R_MIPS_HI16 symbol
# MIPSN32R6-NEXT: 24 21 00 00  addiu  $1, $1, 0
# MIPSN32R6-NEXT:            R_MIPS_LO16 symbol
# MIPSN32R6-NEXT: 00 23 08 21  addu   $1, $1, $3
# MIPSN32R6-NEXT: 7c 2c 00 26  sc     $12, 0($1)

# MIPS64:         3c 01 00 00  lui    $1, 0
# MIPS64-NEXT:               R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    64 21 00 00  daddiu $1, $1, 0
# MIPS64-NEXT:               R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    00 01 0c 38  dsll   $1, $1, 16
# MIPS64-NEXT:    64 21 00 00  daddiu $1, $1, 0
# MIPS64-NEXT:               R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol
# MIPS64-NEXT:    00 01 0c 38  dsll   $1, $1, 16
# MIPS64-NEXT:    00 23 08 2d  daddu  $1, $1, $3
# MIPS64-NEXT:    e0 2c 00 00  sc     $12, 0($1)
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
# MIPS64R6-NEXT:  7c 2c 00 26  sc     $12, 0($1)

sc $12, symbol+8
# MIPS32:         3c 01 00 00  lui    $1, 0
# MIPS32-NEXT:               R_MIPS_HI16  symbol
# MIPS32-NEXT:    e0 2c 00 08  sc     $12, 8($1)
# MIPS32-NEXT:               R_MIPS_LO16  symbol

# MIPS32R6:       3c 01 00 00  aui    $1, $zero, 0
# MIPS32R6-NEXT:             R_MIPS_HI16 symbol
# MIPS32R6-NEXT:  24 21 00 08  addiu  $1, $1, 8
# MIPS32R6-NEXT:             R_MIPS_LO16 symbol
# MIPS32R6-NEXT:  7c 2c 00 26  sc     $12, 0($1)

# MIPSN32:        3c 01 00 00  lui    $1, 0
# MIPSN32-NEXT:              R_MIPS_HI16  symbol+0x8
# MIPSN32-NEXT:   e0 2c 00 00  sc     $12, 0($1)
# MIPSN32-NEXT:              R_MIPS_LO16  symbol+0x8

# MIPSN32R6:      3c 01 00 00  aui    $1, $zero, 0
# MIPSN32R6-NEXT:            R_MIPS_HI16 symbol+0x8
# MIPSN32R6-NEXT: 24 21 00 00  addiu  $1, $1, 0
# MIPSN32R6-NEXT:            R_MIPS_LO16 symbol+0x8
# MIPSN32R6-NEXT: 7c 2c 00 26  sc     $12, 0($1)

# MIPS64:         3c 01 00 00  lui    $1, 0
# MIPS64-NEXT:               R_MIPS_HIGHEST/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64-NEXT:    64 21 00 00  daddiu $1, $1, 0
# MIPS64-NEXT:               R_MIPS_HIGHER/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64-NEXT:    00 01 0c 38  dsll   $1, $1, 16
# MIPS64-NEXT:    64 21 00 00  daddiu $1, $1, 0
# MIPS64-NEXT:               R_MIPS_HI16/R_MIPS_NONE/R_MIPS_NONE  symbol+0x8
# MIPS64-NEXT:    00 01 0c 38  dsll   $1, $1, 16
# MIPS64-NEXT:    e0 2c 00 00  sc     $12, 0($1)
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
# MIPS64R6-NEXT:  7c 2c 00 26  sc     $12, 0($1)

.option pic2

sc $12, symbol
# MIPS32:         8f 81 00 00  lw     $1, 0($gp)
# MIPS32-NEXT:               R_MIPS_GOT16 symbol
# MIPS32-NEXT:    e0 2c 00 00  sc     $12, 0($1)

# MIPS32R6:       8f 81 00 00  lw     $1, 0($gp)
# MIPS32R6-NEXT:             R_MIPS_GOT16 symbol
# MIPS32R6-NEXT:  7c 2c 00 26  sc     $12, 0($1)

# MIPSN32:        8f 81 00 00  lw     $1, 0($gp)
# MIPSN32-NEXT:              R_MIPS_GOT_DISP symbol
# MIPSN32-NEXT:   e0 2c 00 00  sc     $12, 0($1)

# MIPSN32R6:      8f 81 00 00  lw     $1, 0($gp)
# MIPSN32R6-NEXT:            R_MIPS_GOT_DISP symbol
# MIPSN32R6-NEXT: 7c 2c 00 26  sc     $12, 0($1)

# MIPS64:         df 81 00 00  ld     $1, 0($gp)
# MIPS64-NEXT:               R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE symbol
# MIPS64-NEXT:    e0 2c 00 00  sc     $12, 0($1)

# MIPS64R6:       df 81 00 00  ld     $1, 0($gp)
# MIPS64R6-NEXT:             R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE symbol
# MIPS64R6-NEXT:  7c 2c 00 26  sc     $12, 0($1)

sc $12, symbol+8
# MIPS32:         8f 81 00 00  lw     $1, 0($gp)
# MIPS32-NEXT:               R_MIPS_GOT16 symbol
# MIPS32-NEXT:    e0 2c 00 08  sc     $12, 8($1)

# MIPS32R6:       8f 81 00 00  lw     $1, 0($gp)
# MIPS32R6-NEXT:             R_MIPS_GOT16 symbol
# MIPS32R6-NEXT:  24 21 00 08  addiu  $1, $1, 8
# MIPS32R6-NEXT:  7c 2c 00 26  sc     $12, 0($1)

# MIPSN32:        8f 81 00 00  lw     $1, 0($gp)
# MIPSN32-NEXT:              R_MIPS_GOT_DISP symbol
# MIPSN32-NEXT:   e0 2c 00 08  sc     $12, 8($1)

# MIPSN32R6:      8f 81 00 00  lw     $1, 0($gp)
# MIPSN32R6-NEXT:            R_MIPS_GOT_DISP symbol
# MIPSN32R6-NEXT: 24 21 00 08  addiu  $1, $1, 8
# MIPSN32R6-NEXT: 7c 2c 00 26  sc     $12, 0($1)

# MIPS64:         df 81 00 00  ld     $1, 0($gp)
# MIPS64-NEXT:               R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE symbol
# MIPS64-NEXT:    e0 2c 00 08  sc     $12, 8($1)

# MIPS64R6:       df 81 00 00  ld     $1, 0($gp)
# MIPS64R6-NEXT:             R_MIPS_GOT_DISP/R_MIPS_NONE/R_MIPS_NONE symbol
# MIPS64R6-NEXT:  64 21 00 08  daddiu $1, $1, 8
# MIPS64R6-NEXT:  7c 2c 00 26  sc     $12, 0($1)
