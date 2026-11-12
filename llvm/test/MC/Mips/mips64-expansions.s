# RUN: llvm-mc %s -triple=mips64el-unknown-linux -mcpu=mips64r2 -mattr=-xgot \
# RUN:            -show-encoding | FileCheck --check-prefixes=CHECK,GOT %s
# RUN: llvm-mc %s -triple=mips64el-unknown-linux -mcpu=mips64r2 -mattr=+xgot \
# RUN:            -show-encoding | FileCheck --check-prefixes=CHECK,XGOT %s

# Check that signed negative 32-bit immediates are loaded correctly:
  li $10, ~(0x101010)
# CHECK: lui $10, 65519        # encoding: [0xef,0xff,0x0a,0x3c]
# CHECK: ori $10, $10, 61423   # encoding: [0xef,0xef,0x4a,0x35]
# CHECK-NOT: dsll

# Test bne with an immediate as the 2nd operand.
  bne $2, 0x100010001, 1332
# CHECK: addiu $1, $zero, 1         # encoding: [0x01,0x00,0x01,0x24]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 1             # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 1             # encoding: [0x01,0x00,0x21,0x34]
# CHECK: bne  $2, $1, 1332          # encoding: [0x4d,0x01,0x41,0x14]
# CHECK: nop                        # encoding: [0x00,0x00,0x00,0x00]

  bne $2, 0x1000100010001, 1332
# CHECK: lui  $1, 1                 # encoding: [0x01,0x00,0x01,0x3c]
# CHECK: ori  $1, $1, 1             # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 1             # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 1             # encoding: [0x01,0x00,0x21,0x34]
# CHECK: bne  $2, $1, 1332          # encoding: [0x4d,0x01,0x41,0x14]
# CHECK: nop                        # encoding: [0x00,0x00,0x00,0x00]

  bne $2, -0x100010001, 1332
# CHECK: addiu $1, $zero, -2        # encoding: [0xfe,0xff,0x01,0x24]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 65534         # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 65535         # encoding: [0xff,0xff,0x21,0x34]
# CHECK: bne  $2, $1, 1332          # encoding: [0x4d,0x01,0x41,0x14]
# CHECK: nop                        # encoding: [0x00,0x00,0x00,0x00]

  bne $2, -0x1000100010001, 1332
# CHECK: lui  $1, 65534             # encoding: [0xfe,0xff,0x01,0x3c]
# CHECK: ori  $1, $1, 65534         # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 65534         # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 65535         # encoding: [0xff,0xff,0x21,0x34]
# CHECK: bne  $2, $1, 1332          # encoding: [0x4d,0x01,0x41,0x14]
# CHECK: nop                        # encoding: [0x00,0x00,0x00,0x00]

# Test beq with an immediate as the 2nd operand.
  beq $2, 0x100010001, 1332
# CHECK: addiu $1, $zero, 1         # encoding: [0x01,0x00,0x01,0x24]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 1             # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 1             # encoding: [0x01,0x00,0x21,0x34]
# CHECK: beq  $2, $1, 1332          # encoding: [0x4d,0x01,0x41,0x10]
# CHECK: nop                        # encoding: [0x00,0x00,0x00,0x00]

  beq $2, 0x1000100010001, 1332
# CHECK: lui  $1, 1                 # encoding: [0x01,0x00,0x01,0x3c]
# CHECK: ori  $1, $1, 1             # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 1             # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 1             # encoding: [0x01,0x00,0x21,0x34]
# CHECK: beq  $2, $1, 1332          # encoding: [0x4d,0x01,0x41,0x10]
# CHECK: nop                        # encoding: [0x00,0x00,0x00,0x00]

  beq $2, -0x100010001, 1332
# CHECK: addiu $1, $zero, -2        # encoding: [0xfe,0xff,0x01,0x24]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 65534         # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 65535         # encoding: [0xff,0xff,0x21,0x34]
# CHECK: beq  $2, $1, 1332          # encoding: [0x4d,0x01,0x41,0x10]
# CHECK: nop                        # encoding: [0x00,0x00,0x00,0x00]

  beq $2, -0x1000100010001, 1332
# CHECK: lui  $1, 65534             # encoding: [0xfe,0xff,0x01,0x3c]
# CHECK: ori  $1, $1, 65534         # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 65534         # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 65535         # encoding: [0xff,0xff,0x21,0x34]
# CHECK: beq  $2, $1, 1332          # encoding: [0x4d,0x01,0x41,0x10]
# CHECK: nop                        # encoding: [0x00,0x00,0x00,0x00]

# Test one with a symbol in the third operand.
sym:
  bne $2, 0x100010001, sym
# CHECK: addiu $1, $zero, 1         # encoding: [0x01,0x00,0x01,0x24]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 1             # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll $1, $1, 16            # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 1             # encoding: [0x01,0x00,0x21,0x34]
# CHECK: bne  $2, $1, sym           # encoding: [A,A,0x41,0x14]
# CHECK: nop                        # encoding: [0x00,0x00,0x00,0x00]

# Test ulhu with 64-bit immediate addresses.
  ulhu $8, 0x100010001
# CHECK: addiu $1, $zero, 1    # encoding: [0x01,0x00,0x01,0x24]
# CHECK: ori  $1, $1, 1        # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll $1, $1, 16       # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 1        # encoding: [0x01,0x00,0x21,0x34]
# CHECK: lbu  $8, 1($1)        # encoding: [0x01,0x00,0x28,0x90]
# CHECK: lbu  $1, 0($1)        # encoding: [0x00,0x00,0x21,0x90]
# CHECK: sll  $8, $8, 8        # encoding: [0x00,0x42,0x08,0x00]
# CHECK: or   $8, $8, $1       # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, 0x1000100010001
# CHECK: lui  $1, 1            # encoding: [0x01,0x00,0x01,0x3c]
# CHECK: ori  $1, $1, 1        # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll $1, $1, 16       # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 1        # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll $1, $1, 16       # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 1        # encoding: [0x01,0x00,0x21,0x34]
# CHECK: lbu  $8, 1($1)        # encoding: [0x01,0x00,0x28,0x90]
# CHECK: lbu  $1, 0($1)        # encoding: [0x00,0x00,0x21,0x90]
# CHECK: sll  $8, $8, 8        # encoding: [0x00,0x42,0x08,0x00]
# CHECK: or   $8, $8, $1       # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, -0x100010001
# CHECK: addiu $1, $zero, -2   # encoding: [0xfe,0xff,0x01,0x24]
# CHECK: dsll $1, $1, 16       # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 65534    # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll $1, $1, 16       # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 65535    # encoding: [0xff,0xff,0x21,0x34]
# CHECK: lbu  $8, 1($1)        # encoding: [0x01,0x00,0x28,0x90]
# CHECK: lbu  $1, 0($1)        # encoding: [0x00,0x00,0x21,0x90]
# CHECK: sll  $8, $8, 8        # encoding: [0x00,0x42,0x08,0x00]
# CHECK: or   $8, $8, $1       # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, -0x1000100010001
# CHECK: lui  $1, 65534        # encoding: [0xfe,0xff,0x01,0x3c]
# CHECK: ori  $1, $1, 65534    # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll $1, $1, 16       # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 65534    # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll $1, $1, 16       # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 65535    # encoding: [0xff,0xff,0x21,0x34]
# CHECK: lbu  $8, 1($1)        # encoding: [0x01,0x00,0x28,0x90]
# CHECK: lbu  $1, 0($1)        # encoding: [0x00,0x00,0x21,0x90]
# CHECK: sll  $8, $8, 8        # encoding: [0x00,0x42,0x08,0x00]
# CHECK: or   $8, $8, $1       # encoding: [0x25,0x40,0x01,0x01]

# Test ulhu with source register and 64-bit immediate offset.
  ulhu $8, 0x100010001($9)
# CHECK: addiu $1, $zero, 1    # encoding: [0x01,0x00,0x01,0x24]
# CHECK: ori   $1, $1, 1       # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll  $1, $1, 16      # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori   $1, $1, 1       # encoding: [0x01,0x00,0x21,0x34]
# CHECK: daddu $1, $1, $9      # encoding: [0x2d,0x08,0x29,0x00]
# CHECK: lbu   $8, 1($1)       # encoding: [0x01,0x00,0x28,0x90]
# CHECK: lbu   $1, 0($1)       # encoding: [0x00,0x00,0x21,0x90]
# CHECK: sll   $8, $8, 8       # encoding: [0x00,0x42,0x08,0x00]
# CHECK: or    $8, $8, $1      # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, 0x1000100010001($9)
# CHECK: lui   $1, 1           # encoding: [0x01,0x00,0x01,0x3c]
# CHECK: ori   $1, $1, 1       # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll  $1, $1, 16      # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori   $1, $1, 1       # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll  $1, $1, 16      # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori   $1, $1, 1       # encoding: [0x01,0x00,0x21,0x34]
# CHECK: daddu $1, $1, $9      # encoding: [0x2d,0x08,0x29,0x00]
# CHECK: lbu   $8, 1($1)       # encoding: [0x01,0x00,0x28,0x90]
# CHECK: lbu   $1, 0($1)       # encoding: [0x00,0x00,0x21,0x90]
# CHECK: sll   $8, $8, 8       # encoding: [0x00,0x42,0x08,0x00]
# CHECK: or    $8, $8, $1      # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, -0x100010001($9)
# CHECK: addiu $1, $zero, -2   # encoding: [0xfe,0xff,0x01,0x24]
# CHECK: dsll  $1, $1, 16      # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori   $1, $1, 65534   # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll  $1, $1, 16      # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: daddu $1, $1, $9      # encoding: [0x2d,0x08,0x29,0x00]
# CHECK: lbu   $8, 1($1)       # encoding: [0x01,0x00,0x28,0x90]
# CHECK: lbu   $1, 0($1)       # encoding: [0x00,0x00,0x21,0x90]
# CHECK: sll   $8, $8, 8       # encoding: [0x00,0x42,0x08,0x00]
# CHECK: or    $8, $8, $1      # encoding: [0x25,0x40,0x01,0x01]

  ulhu $8, -0x1000100010001($9)
# CHECK: lui   $1, 65534       # encoding: [0xfe,0xff,0x01,0x3c]
# CHECK: ori   $1, $1, 65534   # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll  $1, $1, 16      # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori   $1, $1, 65534   # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll  $1, $1, 16      # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori   $1, $1, 65535   # encoding: [0xff,0xff,0x21,0x34]
# CHECK: daddu $1, $1, $9      # encoding: [0x2d,0x08,0x29,0x00]
# CHECK: lbu   $8, 1($1)       # encoding: [0x01,0x00,0x28,0x90]
# CHECK: lbu   $1, 0($1)       # encoding: [0x00,0x00,0x21,0x90]
# CHECK: sll   $8, $8, 8       # encoding: [0x00,0x42,0x08,0x00]
# CHECK: or    $8, $8, $1      # encoding: [0x25,0x40,0x01,0x01]

# Test ulw with 64-bit immediate addresses.
  ulw $8, 0x100010001
# CHECK: addiu $1, $zero, 1   # encoding: [0x01,0x00,0x01,0x24]
# CHECK: ori  $1, $1, 1       # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll $1, $1, 16      # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 1       # encoding: [0x01,0x00,0x21,0x34]
# CHECK: lwl  $8, 3($1)       # encoding: [0x03,0x00,0x28,0x88]
# CHECK: lwr  $8, 0($1)       # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, 0x1000100010001
# CHECK: lui  $1, 1           # encoding: [0x01,0x00,0x01,0x3c]
# CHECK: ori  $1, $1, 1       # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll $1, $1, 16      # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 1       # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll $1, $1, 16      # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 1       # encoding: [0x01,0x00,0x21,0x34]
# CHECK: lwl  $8, 3($1)       # encoding: [0x03,0x00,0x28,0x88]
# CHECK: lwr  $8, 0($1)       # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, -0x100010001
# CHECK: addiu $1, $zero, -2  # encoding: [0xfe,0xff,0x01,0x24]
# CHECK: dsll $1, $1, 16      # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 65534   # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll $1, $1, 16      # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 65535   # encoding: [0xff,0xff,0x21,0x34]
# CHECK: lwl  $8, 3($1)       # encoding: [0x03,0x00,0x28,0x88]
# CHECK: lwr  $8, 0($1)       # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, -0x1000100010001
# CHECK: lui  $1, 65534       # encoding: [0xfe,0xff,0x01,0x3c]
# CHECK: ori  $1, $1, 65534   # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll $1, $1, 16      # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 65534   # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll $1, $1, 16      # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori  $1, $1, 65535   # encoding: [0xff,0xff,0x21,0x34]
# CHECK: lwl  $8, 3($1)       # encoding: [0x03,0x00,0x28,0x88]
# CHECK: lwr  $8, 0($1)       # encoding: [0x00,0x00,0x28,0x98]

# Test ulw with source register and 64-bit immediate offset.
  ulw $8, 0x100010001($9)
# CHECK: addiu $1, $zero, 1   # encoding: [0x01,0x00,0x01,0x24]
# CHECK: ori   $1, $1, 1      # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll  $1, $1, 16     # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori   $1, $1, 1      # encoding: [0x01,0x00,0x21,0x34]
# CHECK: daddu $1, $1, $9     # encoding: [0x2d,0x08,0x29,0x00]
# CHECK: lwl   $8, 3($1)      # encoding: [0x03,0x00,0x28,0x88]
# CHECK: lwr   $8, 0($1)      # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, 0x1000100010001($9)
# CHECK: lui   $1, 1          # encoding: [0x01,0x00,0x01,0x3c]
# CHECK: ori   $1, $1, 1      # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll  $1, $1, 16     # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori   $1, $1, 1      # encoding: [0x01,0x00,0x21,0x34]
# CHECK: dsll  $1, $1, 16     # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori   $1, $1, 1      # encoding: [0x01,0x00,0x21,0x34]
# CHECK: daddu $1, $1, $9     # encoding: [0x2d,0x08,0x29,0x00]
# CHECK: lwl   $8, 3($1)      # encoding: [0x03,0x00,0x28,0x88]
# CHECK: lwr   $8, 0($1)      # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, -0x100010001($9)
# CHECK: addiu $1, $zero, -2  # encoding: [0xfe,0xff,0x01,0x24]
# CHECK: dsll  $1, $1, 16     # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori   $1, $1, 65534  # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll  $1, $1, 16     # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori   $1, $1, 65535  # encoding: [0xff,0xff,0x21,0x34]
# CHECK: daddu $1, $1, $9     # encoding: [0x2d,0x08,0x29,0x00]
# CHECK: lwl   $8, 3($1)      # encoding: [0x03,0x00,0x28,0x88]
# CHECK: lwr   $8, 0($1)      # encoding: [0x00,0x00,0x28,0x98]

  ulw $8, -0x1000100010001($9)
# CHECK: lui   $1, 65534      # encoding: [0xfe,0xff,0x01,0x3c]
# CHECK: ori   $1, $1, 65534  # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll  $1, $1, 16     # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori   $1, $1, 65534  # encoding: [0xfe,0xff,0x21,0x34]
# CHECK: dsll  $1, $1, 16     # encoding: [0x38,0x0c,0x01,0x00]
# CHECK: ori   $1, $1, 65535  # encoding: [0xff,0xff,0x21,0x34]
# CHECK: daddu $1, $1, $9     # encoding: [0x2d,0x08,0x29,0x00]
# CHECK: lwl   $8, 3($1)      # encoding: [0x03,0x00,0x28,0x88]
# CHECK: lwr   $8, 0($1)      # encoding: [0x00,0x00,0x28,0x98]

# Test lb/sb/ld/sd/lld with offsets exceeding 16-bits in size.

    ld  $4, 0x8000
# CHECK:      lui     $4, 1
# CHECK-NEXT: ld      $4, -32768($4)

    ld  $4, 0x20008($3)
# CHECK:      lui     $4, 2
# CHECK-NEXT: addu    $4, $4, $3
# CHECK-NEXT: ld      $4, 8($4)

    ld  $4,0x100010004
# CHECK:      addiu   $4, $zero, 1
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ori     $4, $4, 1
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ld      $4, 4($4)

    ld  $4,0x1800180018004
# CHECK:      lui     $4, 1
# CHECK-NEXT: ori     $4, $4, 32769
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ori     $4, $4, 32770
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ld      $4, -32764($4)

    ld  $4,0x1800180018004($3)
# CHECK:      lui     $4, 1
# CHECK-NEXT: ori     $4, $4, 32769
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ori     $4, $4, 32770
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: daddu   $4, $4, $3
# CHECK-NEXT: ld      $4, -32764($4)

    sd  $4, 0x8000
# CHECK:      lui     $1, 1
# CHECK-NEXT: sd      $4, -32768($1)

    sd  $4, 0x20008($3)
# CHECK:      lui     $1, 2
# CHECK-NEXT: addu    $1, $1, $3
# CHECK-NEXT: sd      $4, 8($1)

    sd  $4,0x100010004
# CHECK:      addiu   $1, $zero, 1
# CHECK-NEXT: dsll    $1, $1, 16
# CHECK-NEXT: ori     $1, $1, 1
# CHECK-NEXT: dsll    $1, $1, 16
# CHECK-NEXT: sd      $4, 4($1)

    sd  $4,0x1800180018004
# CHECK:      lui     $1, 1
# CHECK-NEXT: ori     $1, $1, 32769
# CHECK-NEXT: dsll    $1, $1, 16
# CHECK-NEXT: ori     $1, $1, 32770
# CHECK-NEXT: dsll    $1, $1, 16
# CHECK-NEXT: sd      $4, -32764($1)

    sd  $4,0x1800180018004($3)
# CHECK:      lui     $1, 1
# CHECK-NEXT: ori     $1, $1, 32769
# CHECK-NEXT: dsll    $1, $1, 16
# CHECK-NEXT: ori     $1, $1, 32770
# CHECK-NEXT: dsll    $1, $1, 16
# CHECK-NEXT: daddu   $1, $1, $3
# CHECK-NEXT: sd      $4, -32764($1)

    lld $4, 0x8000
# CHECK:      lui     $4, 1
# CHECK-NEXT: lld     $4, -32768($4)

    lld $4, 0x20008($3)
# CHECK:      lui     $4, 2
# CHECK-NEXT: addu    $4, $4, $3
# CHECK-NEXT: lld     $4, 8($4)

    lld $4,0x100010004
# CHECK:      addiu   $4, $zero, 1
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ori     $4, $4, 1
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: lld     $4, 4($4)

    lld $4,0x1800180018004
# CHECK:      lui     $4, 1
# CHECK-NEXT: ori     $4, $4, 32769
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ori     $4, $4, 32770
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: lld     $4, -32764($4)

    lld $4,0x1800180018004($3)
# CHECK:      lui     $4, 1
# CHECK-NEXT: ori     $4, $4, 32769
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ori     $4, $4, 32770
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: daddu   $4, $4, $3
# CHECK-NEXT: lld     $4, -32764($4)

    lb  $4,0x100010004
# CHECK:      addiu   $4, $zero, 1
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ori     $4, $4, 1
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: lb      $4, 4($4)

    lb  $4,0x1800180018004
# CHECK:      lui     $4, 1
# CHECK-NEXT: ori     $4, $4, 32769
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ori     $4, $4, 32770
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: lb      $4, -32764($4)

    lb  $4,0x1800180018004($3)
# CHECK:      lui     $4, 1
# CHECK-NEXT: ori     $4, $4, 32769
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ori     $4, $4, 32770
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: daddu   $4, $4, $3
# CHECK-NEXT: lb      $4, -32764($4)

    sb  $4,0x100010004
# CHECK:      addiu   $1, $zero, 1
# CHECK-NEXT: dsll    $1, $1, 16
# CHECK-NEXT: ori     $1, $1, 1
# CHECK-NEXT: dsll    $1, $1, 16
# CHECK-NEXT: sb      $4, 4($1)

    sb  $4,0x1800180018004
# CHECK:      lui     $1, 1
# CHECK-NEXT: ori     $1, $1, 32769
# CHECK-NEXT: dsll    $1, $1, 16
# CHECK-NEXT: ori     $1, $1, 32770
# CHECK-NEXT: dsll    $1, $1, 16
# CHECK-NEXT: sb      $4, -32764($1)

    sb  $4,0x1800180018004($3)
# CHECK:      lui     $1, 1
# CHECK-NEXT: ori     $1, $1, 32769
# CHECK-NEXT: dsll    $1, $1, 16
# CHECK-NEXT: ori     $1, $1, 32770
# CHECK-NEXT: dsll    $1, $1, 16
# CHECK-NEXT: daddu   $1, $1, $3
# CHECK-NEXT: sb      $4, -32764($1)

    lh  $4,0x100010004
# CHECK:      addiu   $4, $zero, 1
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ori     $4, $4, 1
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: lh      $4, 4($4)

    lh  $4,0x1800180018004
# CHECK:      lui     $4, 1
# CHECK-NEXT: ori     $4, $4, 32769
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ori     $4, $4, 32770
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: lh      $4, -32764($4)

    lh  $4,0x1800180018004($3)
# CHECK:      lui     $4, 1
# CHECK-NEXT: ori     $4, $4, 32769
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ori     $4, $4, 32770
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: daddu   $4, $4, $3
# CHECK-NEXT: lh      $4, -32764($4)

    lhu $4,0x100010004
# CHECK:      addiu   $4, $zero, 1
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ori     $4, $4, 1
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: lhu     $4, 4($4)

    lhu $4,0x1800180018004
# CHECK:      lui     $4, 1
# CHECK-NEXT: ori     $4, $4, 32769
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ori     $4, $4, 32770
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: lhu     $4, -32764($4)

    lhu $4,0x1800180018004($3)
# CHECK:      lui     $4, 1
# CHECK-NEXT: ori     $4, $4, 32769
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: ori     $4, $4, 32770
# CHECK-NEXT: dsll    $4, $4, 16
# CHECK-NEXT: daddu   $4, $4, $3
# CHECK-NEXT: lhu     $4, -32764($4)

# LW/SW and LDC1/SDC1 of symbol address, done by MipsAsmParser::expandMemInst():
  .option pic2
  lw $10, symbol($4)
# GOT:      ld      $10, %got_disp(symbol)($gp) # encoding: [A,A,0x8a,0xdf]
# GOT-NEXT:                                     #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# GOT-NEXT: daddu   $10, $10, $4                # encoding: [0x2d,0x50,0x44,0x01]
# GOT-NEXT: lw      $10, 0($10)                 # encoding: [0x00,0x00,0x4a,0x8d]

# XGOT:      lui    $10, %got_hi(symbol)        # encoding: [A,A,0x0a,0x3c]
# XGOT-NEXT:                                    #   fixup A - offset: 0, value: %got_hi(symbol), kind: fixup_Mips_GOT_HI16
# XGOT-NEXT: daddu  $10, $10, $gp               # encoding: [0x2d,0x50,0x5c,0x01]
# XGOT-NEXT: ld     $10, %got_lo(symbol)($10)   # encoding: [A,A,0x4a,0xdd]
# XGOT-NEXT:                                    #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT-NEXT: daddu  $10, $10, $4                # encoding: [0x2d,0x50,0x44,0x01]
# XGOT-NEXT: lw     $10, 0($10)                 # encoding: [0x00,0x00,0x4a,0x8d]

  sw $10, symbol($9)
# GOT:      ld      $1, %got_disp(symbol)($gp)  # encoding: [A,A,0x81,0xdf]
# GOT-NEXT:                                     #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# GOT-NEXT: daddu   $1, $1, $9                  # encoding: [0x2d,0x08,0x29,0x00]
# GOT-NEXT: sw      $10, 0($1)                  # encoding: [0x00,0x00,0x2a,0xac]

# XGOT:      lui    $1, %got_hi(symbol)         # encoding: [A,A,0x01,0x3c]
# XGOT-NEXT:                                    #   fixup A - offset: 0, value: %got_hi(symbol), kind: fixup_Mips_GOT_HI16
# XGOT-NEXT: daddu  $1, $1, $gp                 # encoding: [0x2d,0x08,0x3c,0x00]
# XGOT-NEXT: ld     $1, %got_lo(symbol)($1)     # encoding: [A,A,0x21,0xdc]
# XGOT-NEXT:                                    #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT-NEXT: daddu  $1, $1, $9                  # encoding: [0x2d,0x08,0x29,0x00]
# XGOT-NEXT: sw     $10, 0($1)                  # encoding: [0x00,0x00,0x2a,0xac]

  lw $8, sym+8
# GOT:      ld      $8, %got_disp(sym)($gp)     # encoding: [A,A,0x88,0xdf]
# GOT-NEXT:                                     #   fixup A - offset: 0, value: %got_disp(sym), kind: fixup_Mips_GOT_DISP
# GOT-NEXT: lw      $8, 8($8)                   # encoding: [0x08,0x00,0x08,0x8d]

# XGOT:      ld     $8, %got_disp(sym)($gp)     # encoding: [A,A,0x88,0xdf]
# XGOT-NEXT:                                    #   fixup A - offset: 0, value: %got_disp(sym), kind: fixup_Mips_GOT_DISP
# XGOT-NEXT: lw     $8, 8($8)                   # encoding: [0x08,0x00,0x08,0x8d]

  sw $8, sym+8
# GOT:      ld      $1, %got_disp(sym)($gp)     # encoding: [A,A,0x81,0xdf]
# GOT-NEXT:                                     #   fixup A - offset: 0, value: %got_disp(sym), kind: fixup_Mips_GOT_DISP
# GOT-NEXT: sw      $8, 8($1)                   # encoding: [0x08,0x00,0x28,0xac]

# XGOT:      ld     $1, %got_disp(sym)($gp)     # encoding: [A,A,0x81,0xdf]
# XGOT-NEXT:                                    #   fixup A - offset: 0, value: %got_disp(sym), kind: fixup_Mips_GOT_DISP
# XGOT-NEXT: sw     $8, 8($1)                   # encoding: [0x08,0x00,0x28,0xac]

  lw $10, 655483($4)
# GOT:      lui     $10, 10                     # encoding: [0x0a,0x00,0x0a,0x3c]
# GOT-NEXT: daddu   $10, $10, $4                # encoding: [0x2d,0x50,0x44,0x01]
# GOT-NEXT: lw      $10, 123($10)               # encoding: [0x7b,0x00,0x4a,0x8d]

# XGOT:      lui    $10, 10                     # encoding: [0x0a,0x00,0x0a,0x3c]
# XGOT-NEXT: daddu  $10, $10, $4                # encoding: [0x2d,0x50,0x44,0x01]
# XGOT-NEXT: lw     $10, 123($10)               # encoding: [0x7b,0x00,0x4a,0x8d]
  sw $10, 123456($9)
# GOT:      lui     $1, 2                       # encoding: [0x02,0x00,0x01,0x3c]
# GOT-NEXT: daddu   $1, $1, $9                  # encoding: [0x2d,0x08,0x29,0x00]
# GOT-NEXT: sw      $10, -7616($1)              # encoding: [0x40,0xe2,0x2a,0xac]

# XGOT:      lui    $1, 2                       # encoding: [0x02,0x00,0x01,0x3c]
# XGOT-NEXT: daddu  $1, $1, $9                  # encoding: [0x2d,0x08,0x29,0x00]
# XGOT-NEXT: sw     $10, -7616($1)              # encoding: [0x40,0xe2,0x2a,0xac]

  lw $8, symbol+8
# GOT:      ld      $8, %got_disp(symbol)($gp)  # encoding: [A,A,0x88,0xdf]
# GOT-NEXT:                                     #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# GOT-NEXT: lw      $8, 8($8)                   # encoding: [0x08,0x00,0x08,0x8d]

# XGOT:      lui    $8, %got_hi(symbol)         # encoding: [A,A,0x08,0x3c]
# XGOT-NEXT:                                    #   fixup A - offset: 0, value: %got_hi(symbol), kind: fixup_Mips_GOT_HI16
# XGOT-NEXT: daddu  $8, $8, $gp                 # encoding: [0x2d,0x40,0x1c,0x01]
# XGOT-NEXT: ld     $8, %got_lo(symbol)($8)     # encoding: [A,A,0x08,0xdd]
# XGOT-NEXT:                                    #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT-NEXT: lw     $8, 8($8)                   # encoding: [0x08,0x00,0x08,0x8d]

  sw $8, symbol+8
# GOT:      ld      $1, %got_disp(symbol)($gp)  # encoding: [A,A,0x81,0xdf]
# GOT-NEXT:                                     #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# GOT-NEXT: sw $8, 8($1)                        # encoding: [0x08,0x00,0x28,0xac]

# XGOT:      lui    $1, %got_hi(symbol)         # encoding: [A,A,0x01,0x3c]
# XGOT-NEXT:                                    #   fixup A - offset: 0, value: %got_hi(symbol), kind: fixup_Mips_GOT_HI16
# XGOT-NEXT: daddu  $1, $1, $gp                 # encoding: [0x2d,0x08,0x3c,0x00]
# XGOT-NEXT: ld     $1, %got_lo(symbol)($1)     # encoding: [A,A,0x21,0xdc]
# XGOT-NEXT:                                    #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT-NEXT: sw     $8, 8($1)                   # encoding: [0x08,0x00,0x28,0xac]

  ldc1 $f0, symbol
# GOT:      ld      $1, %got_disp(symbol)($gp) # encoding: [A,A,0x81,0xdf]
# GOT-NEXT:                                    #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# GOT-NEXT: ldc1    $f0, 0($1)                 # encoding: [0x00,0x00,0x20,0xd4]

# XGOT:      lui    $1, %got_hi(symbol)         # encoding: [A,A,0x01,0x3c]
# XGOT-NEXT:                                    #   fixup A - offset: 0, value: %got_hi(symbol), kind: fixup_Mips_GOT_HI16
# XGOT-NEXT: daddu  $1, $1, $gp                 # encoding: [0x2d,0x08,0x3c,0x00]
# XGOT-NEXT: ld     $1, %got_lo(symbol)($1)     # encoding: [A,A,0x21,0xdc]
# XGOT-NEXT:                                    #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT-NEXT: ldc1   $f0, 0($1)                  # encoding: [0x00,0x00,0x20,0xd4]

  sdc1 $f0, symbol
# GOT:      ld      $1, %got_disp(symbol)($gp) # encoding: [A,A,0x81,0xdf]
# GOT-NEXT:                                    #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
# GOT-NEXT: sdc1    $f0, 0($1)                 # encoding: [0x00,0x00,0x20,0xf4]

# XGOT:      lui    $1, %got_hi(symbol)         # encoding: [A,A,0x01,0x3c]
# XGOT-NEXT:                                    #   fixup A - offset: 0, value: %got_hi(symbol), kind: fixup_Mips_GOT_HI16
# XGOT-NEXT: daddu  $1, $1, $gp                 # encoding: [0x2d,0x08,0x3c,0x00]
# XGOT-NEXT: ld     $1, %got_lo(symbol)($1)     # encoding: [A,A,0x21,0xdc]
# XGOT-NEXT:                                    #   fixup A - offset: 0, value: %got_lo(symbol), kind: fixup_Mips_GOT_LO16
# XGOT-NEXT: sdc1   $f0, 0($1)                  # encoding: [0x00,0x00,0x20,0xf4]
