# RUN: llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips | \
# RUN: FileCheck -check-prefix=CHECK-EL %s
# RUN: llvm-mc %s -triple=mips -show-encoding -mattr=micromips | \
# RUN: FileCheck -check-prefix=CHECK-EB %s
# Check that the assembler can handle the documented syntax
# for arithmetic and logical instructions.
#------------------------------------------------------------------------------
# MicroMIPS 16-bit Instructions
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL: addu16  $6, $17, $4     # encoding: [0x42,0x07]
# CHECK-EL: subu16  $5, $16, $3     # encoding: [0xb1,0x06]
# CHECK-EL: andi16  $16, $2, 31     # encoding: [0x29,0x2c]
# CHECK-EL: and16   $16, $2         # encoding: [0x82,0x44]
# CHECK-EL: not16   $17, $3         # encoding: [0x0b,0x44]
# CHECK-EL: or16    $16, $4         # encoding: [0xc4,0x44]
# CHECK-EL: xor16   $17, $5         # encoding: [0x4d,0x44]
# CHECK-EL: sll16   $3, $16, 5      # encoding: [0x8a,0x25]
# CHECK-EL: srl16   $4, $17, 6      # encoding: [0x1d,0x26]
# CHECK-EL: lbu16   $3, 4($17)      # encoding: [0x94,0x09]
# CHECK-EL: lbu16   $3, -1($16)     # encoding: [0x8f,0x09]
# CHECK-EL: lhu16   $3, 4($16)      # encoding: [0x82,0x29]
# CHECK-EL: lw16    $4, 8($17)      # encoding: [0x12,0x6a]
# CHECK-EL: sb16    $3, 4($16)      # encoding: [0x84,0x89]
# CHECK-EL: sh16    $4, 8($17)      # encoding: [0x14,0xaa]
# CHECK-EL: sw16    $4, 4($17)      # encoding: [0x11,0xea]
# CHECK-EL: sw16    $zero, 4($17)   # encoding: [0x11,0xe8]
# CHECK-EL: lw      $3, 32($gp)     # encoding: [0x88,0x65]
# CHECK-EL: lw      $3, 32($sp)     # encoding: [0x68,0x48]
# CHECK-EL: sw      $4, 124($sp)    # encoding: [0x9f,0xc8]
# CHECK-EL: li16    $3, -1          # encoding: [0xff,0xed]
# CHECK-EL: li16    $3, 126         # encoding: [0xfe,0xed]
# CHECK-EL: addiur1sp $7, 4         # encoding: [0x83,0x6f]
# CHECK-EL: addiur2 $6, $7, -1      # encoding: [0x7e,0x6f]
# CHECK-EL: addiur2 $6, $7, 12      # encoding: [0x76,0x6f]
# CHECK-EL: addius5 $7, -2          # encoding: [0xfc,0x4c]
# CHECK-EL: addiusp -1028           # encoding: [0xff,0x4f]
# CHECK-EL: addiusp -1032           # encoding: [0xfd,0x4f]
# CHECK-EL: addiusp 1024            # encoding: [0x01,0x4c]
# CHECK-EL: addiusp 1028            # encoding: [0x03,0x4c]
# CHECK-EL: addiusp -16             # encoding: [0xf9,0x4f]
# CHECK-EL: mfhi    $9              # encoding: [0x09,0x46]
# CHECK-EL: mflo    $9              # encoding: [0x49,0x46]
# CHECK-EL: move    $25, $1         # encoding: [0x21,0x0f]
# CHECK-EL: movep   $5, $6, $2, $3  # encoding: [0x34,0x84]
# CHECK-EL: jrc     $9              # encoding: [0xa9,0x45]
# CHECK-NEXT: jalr    $9            # encoding: [0xc9,0x45]
# CHECK-EL: jraddiusp 20            # encoding: [0x05,0x47]
# CHECK-NEXT: jalrs16 $9            # encoding: [0xe9,0x45]
# CHECK-EL: nop                     # encoding: [0x00,0x0c]
# CHECK-EL: jr16    $9              # encoding: [0x89,0x45]
# CHECK-EL: nop                     # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EL: beqz16 $6, 20           # encoding: [0x0a,0x8f]
# CHECK-EL: nop                     # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EL: bnez16 $6, 20           # encoding: [0x0a,0xaf]
# CHECK-EL: nop                     # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EL: b16 132                 # encoding: [0x42,0xcc]
# CHECK-EL: nop
# CHECK-EL: b16 132                 # encoding: [0x42,0xcc]
# CHECK-EL: nop
# CHECK-EL: break16 8               # encoding: [0x88,0x46]
# CHECK-EL: sdbbp16 14              # encoding: [0xce,0x46]
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB: addu16  $6, $17, $4     # encoding: [0x07,0x42]
# CHECK-EB: subu16  $5, $16, $3     # encoding: [0x06,0xb1]
# CHECK-EB: andi16  $16, $2, 31     # encoding: [0x2c,0x29]
# CHECK-EB: and16   $16, $2         # encoding: [0x44,0x82]
# CHECK-EB: not16   $17, $3         # encoding: [0x44,0x0b]
# CHECK-EB: or16    $16, $4         # encoding: [0x44,0xc4]
# CHECK-EB: xor16   $17, $5         # encoding: [0x44,0x4d]
# CHECK-EB: sll16   $3, $16, 5      # encoding: [0x25,0x8a]
# CHECK-EB: srl16   $4, $17, 6      # encoding: [0x26,0x1d]
# CHECK-EB: lbu16   $3, 4($17)      # encoding: [0x09,0x94]
# CHECK-EB: lbu16   $3, -1($16)     # encoding: [0x09,0x8f]
# CHECK-EB: lhu16   $3, 4($16)      # encoding: [0x29,0x82]
# CHECK-EB: lw16    $4, 8($17)      # encoding: [0x6a,0x12]
# CHECK-EB: sb16    $3, 4($16)      # encoding: [0x89,0x84]
# CHECK-EB: sh16    $4, 8($17)      # encoding: [0xaa,0x14]
# CHECK-EB: sw16    $4, 4($17)      # encoding: [0xea,0x11]
# CHECK-EB: sw16    $zero, 4($17)   # encoding: [0xe8,0x11]
# CHECK-EB: lw      $3, 32($gp)     # encoding: [0x65,0x88]
# CHECK-EB: lw      $3, 32($sp)     # encoding: [0x48,0x68]
# CHECK-EB: sw      $4, 124($sp)    # encoding: [0xc8,0x9f]
# CHECK-EB: li16    $3, -1          # encoding: [0xed,0xff]
# CHECK-EB: li16    $3, 126         # encoding: [0xed,0xfe]
# CHECK-EB: addiur1sp $7, 4         # encoding: [0x6f,0x83]
# CHECK-EB: addiur2 $6, $7, -1      # encoding: [0x6f,0x7e]
# CHECK-EB: addiur2 $6, $7, 12      # encoding: [0x6f,0x76]
# CHECK-EB: addius5 $7, -2          # encoding: [0x4c,0xfc]
# CHECK-EB: addiusp -1028           # encoding: [0x4f,0xff]
# CHECK-EB: addiusp -1032           # encoding: [0x4f,0xfd]
# CHECK-EB: addiusp 1024            # encoding: [0x4c,0x01]
# CHECK-EB: addiusp 1028            # encoding: [0x4c,0x03]
# CHECK-EB: addiusp -16             # encoding: [0x4f,0xf9]
# CHECK-EB: mfhi    $9              # encoding: [0x46,0x09]
# CHECK-EB: mflo    $9              # encoding: [0x46,0x49]
# CHECK-EB: move    $25, $1         # encoding: [0x0f,0x21]
# CHECK-EB: movep   $5, $6, $2, $3  # encoding: [0x84,0x34]
# CHECK-EB: jrc     $9              # encoding: [0x45,0xa9]
# CHECK-NEXT: jalr    $9            # encoding: [0x45,0xc9]
# CHECK-EB: jraddiusp 20            # encoding: [0x47,0x05]
# CHECK-NEXT: jalrs16 $9            # encoding: [0x45,0xe9]
# CHECK-EB: nop                     # encoding: [0x0c,0x00]
# CHECK-EB: jr16    $9              # encoding: [0x45,0x89]
# CHECK-EB: nop                     # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EB: beqz16 $6, 20           # encoding: [0x8f,0x0a]
# CHECK-EB: nop                     # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EB: bnez16 $6, 20           # encoding: [0xaf,0x0a]
# CHECK-EB: nop                     # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EB: b16 132                 # encoding: [0xcc,0x42]
# CHECK-EB: nop
# CHECK-EB: b16 132                 # encoding: [0xcc,0x42]
# CHECK-EB: nop
# CHECK-EB: break16 8               # encoding: [0x46,0x88]
# CHECK-EB: sdbbp16 14              # encoding: [0x46,0xce]

    addu16  $6, $17, $4
    subu16  $5, $16, $3
    andi16  $16, $2, 31
    and16   $16, $2
    not16   $17, $3
    or16    $16, $4
    xor16   $17, $5
    sll16   $3, $16, 5
    srl16   $4, $17, 6
    lbu16   $3, 4($17)
    lbu16   $3, -1($16)
    lhu16   $3, 4($16)
    lw16    $4, 8($17)
    sb16    $3, 4($16)
    sh16    $4, 8($17)
    sw16    $4, 4($17)
    sw16    $0, 4($17)
    lw      $3, 32($gp)
    lw      $3, 32($sp)
    sw      $4, 124($sp)
    li16    $3, -1
    li16    $3, 126
    addiur1sp $7, 4
    addiur2 $6, $7, -1
    addiur2 $6, $7, 12
    addius5 $7, -2
    addiusp -1028
    addiusp -1032
    addiusp 1024
    addiusp 1028
    addiusp -16
    mfhi    $9
    mflo    $9
    move    $25, $1
    movep   $5, $6, $2, $3
    jrc     $9
    jalr    $9
    jraddiusp 20
    jalrs16 $9
    jr16    $9
    beqz16 $6, 20
    bnez16 $6, 20
    b   132
    b16 132
    break16 8
    sdbbp16 14
