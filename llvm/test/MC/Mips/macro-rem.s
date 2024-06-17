# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32 | \
# RUN:   FileCheck %s --check-prefix=CHECK-NOTRAP
# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32 \
# RUN:  -mattr=+use-tcc-in-div | FileCheck %s --check-prefix=CHECK-TRAP

  rem $4,$5
# CHECK-NOTRAP: bnez $5, $tmp0            # encoding: [A,A,0xa0,0x14]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: ($tmp0)-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: div $zero, $4, $5         # encoding: [0x1a,0x00,0x85,0x00]
# CHECK-NOTRAP: break 7                   # encoding: [0x0d,0x00,0x07,0x00]
# CHECK-NOTRAP: $tmp0
# CHECK-NOTRAP: addiu $1, $zero, -1       # encoding: [0xff,0xff,0x01,0x24]
# CHECK-NOTRAP: bne $5, $1, $tmp1         # encoding: [A,A,0xa1,0x14]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: ($tmp1)-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: lui $1, 32768             # encoding: [0x00,0x80,0x01,0x3c]
# CHECK-NOTRAP: bne $4, $1, $tmp1         # encoding: [A,A,0x81,0x14]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: ($tmp1)-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: nop                       # encoding: [0x00,0x00,0x00,0x00]
# CHECK-NOTRAP: break 6                   # encoding: [0x0d,0x00,0x06,0x00]
# CHECK-NOTRAP: $tmp1
# CHECK-NOTRAP: mfhi $4                   # encoding: [0x10,0x20,0x00,0x00]

# CHECK-TRAP: teq $5, $zero, 7            # encoding: [0xf4,0x01,0xa0,0x00]
# CHECK-TRAP: div $zero, $4, $5           # encoding: [0x1a,0x00,0x85,0x00]
# CHECK-TRAP: addiu $1, $zero, -1         # encoding: [0xff,0xff,0x01,0x24]
# CHECK-TRAP: bne $5, $1, $tmp0           # encoding: [A,A,0xa1,0x14]
# CHECK-TRAP: lui $1, 32768               # encoding: [0x00,0x80,0x01,0x3c]
# CHECK-TRAP: teq $4, $1, 6               # encoding: [0xb4,0x01,0x81,0x00]
# CHECK-TRAP: $tmp0
# CHECK-TRAP: mfhi $4                     # encoding: [0x10,0x20,0x00,0x00]

  rem $4,$0
# CHECK-NOTRAP: break 7                   # encoding: [0x0d,0x00,0x07,0x00]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0xf4,0x01,0x00,0x00]

  rem $4,0
# CHECK-NOTRAP: break 7                   # encoding: [0x0d,0x00,0x07,0x00]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0xf4,0x01,0x00,0x00]

  rem $0,0
# CHECK-NOTRAP: break 7                   # encoding: [0x0d,0x00,0x07,0x00]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0xf4,0x01,0x00,0x00]

  rem $4,1
# CHECK-NOTRAP: move $4, $zero            # encoding: [0x25,0x20,0x00,0x00]
# CHECK-TRAP: move $4, $zero              # encoding: [0x25,0x20,0x00,0x00]

  rem $4,-1
# CHECK-NOTRAP: move $4, $zero            # encoding: [0x25,0x20,0x00,0x00]
# CHECK-TRAP: move $4, $zero              # encoding: [0x25,0x20,0x00,0x00]

  rem $4,2
# CHECK-NOTRAP: addiu $1, $zero, 2        # encoding: [0x02,0x00,0x01,0x24]
# CHECK-NOTRAP: div $zero, $4, $1         # encoding: [0x1a,0x00,0x81,0x00]
# CHECK-NOTRAP: mfhi $4                   # encoding: [0x10,0x20,0x00,0x00]
# CHECK-TRAP: addiu $1, $zero, 2          # encoding: [0x02,0x00,0x01,0x24]
# CHECK-TRAP: div $zero, $4, $1           # encoding: [0x1a,0x00,0x81,0x00]
# CHECK-TRAP: mfhi $4                     # encoding: [0x10,0x20,0x00,0x00]

  rem $4,0x8000
# CHECK-NOTRAP: ori $1, $zero, 32768      # encoding: [0x00,0x80,0x01,0x34]
# CHECK-NOTRAP: div $zero, $4, $1         # encoding: [0x1a,0x00,0x81,0x00]
# CHECK-NOTRAP: mfhi $4                   # encoding: [0x10,0x20,0x00,0x00]
# CHECK-TRAP: ori $1, $zero, 32768        # encoding: [0x00,0x80,0x01,0x34]
# CHECK-TRAP: div $zero, $4, $1           # encoding: [0x1a,0x00,0x81,0x00]
# CHECK-TRAP: mfhi $4                     # encoding: [0x10,0x20,0x00,0x00]

  rem $4,-0x8000
# CHECK-NOTRAP: addiu $1, $zero, -32768   # encoding: [0x00,0x80,0x01,0x24]
# CHECK-NOTRAP: div $zero, $4, $1         # encoding: [0x1a,0x00,0x81,0x00]
# CHECK-NOTRAP: mfhi $4                   # encoding: [0x10,0x20,0x00,0x00]
# CHECK-TRAP: addiu $1, $zero, -32768     # encoding: [0x00,0x80,0x01,0x24]
# CHECK-TRAP: div $zero, $4, $1           # encoding: [0x1a,0x00,0x81,0x00]
# CHECK-TRAP: mfhi $4                     # encoding: [0x10,0x20,0x00,0x00]

  rem $4,0x10000
# CHECK-NOTRAP: lui $1, 1                 # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-NOTRAP: div $zero, $4, $1         # encoding: [0x1a,0x00,0x81,0x00]
# CHECK-NOTRAP: mfhi $4                   # encoding: [0x10,0x20,0x00,0x00]
# CHECK-TRAP: lui $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-TRAP: div $zero, $4, $1           # encoding: [0x1a,0x00,0x81,0x00]
# CHECK-TRAP: mfhi $4                     # encoding: [0x10,0x20,0x00,0x00]

  rem $4,0x1a5a5
# CHECK-NOTRAP: lui $1, 1                 # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-NOTRAP: ori $1, $1, 42405         # encoding: [0xa5,0xa5,0x21,0x34]
# CHECK-NOTRAP: div $zero, $4, $1         # encoding: [0x1a,0x00,0x81,0x00]
# CHECK-NOTRAP: mfhi $4                   # encoding: [0x10,0x20,0x00,0x00]
# CHECK-TRAP: lui $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-TRAP: ori $1, $1, 42405           # encoding: [0xa5,0xa5,0x21,0x34]
# CHECK-TRAP: div $zero, $4, $1           # encoding: [0x1a,0x00,0x81,0x00]
# CHECK-TRAP: mfhi $4                     # encoding: [0x10,0x20,0x00,0x00]

  rem $4,$5,$6
# CHECK-NOTRAP: bnez $6, $tmp2            # encoding: [A,A,0xc0,0x14]
# CHECK-NOTRAP: div $zero, $5, $6         # encoding: [0x1a,0x00,0xa6,0x00]
# CHECK-NOTRAP: break 7                   # encoding: [0x0d,0x00,0x07,0x00]
# CHECK-NOTRAP: $tmp2
# CHECK-NOTRAP: addiu $1, $zero, -1       # encoding: [0xff,0xff,0x01,0x24]
# CHECK-NOTRAP: bne $6, $1, $tmp3         # encoding: [A,A,0xc1,0x14]
# CHECK-NOTRAP: lui $1, 32768             # encoding: [0x00,0x80,0x01,0x3c]
# CHECK-NOTRAP: bne $5, $1, $tmp3         # encoding: [A,A,0xa1,0x14]
# CHECK-NOTRAP: nop                       # encoding: [0x00,0x00,0x00,0x00]
# CHECK-NOTRAP: break 6                   # encoding: [0x0d,0x00,0x06,0x00]
# CHECK-NOTRAP: $tmp3
# CHECK-NOTRAP: mfhi $4                   # encoding: [0x10,0x20,0x00,0x00]
# CHECK-TRAP: teq $6, $zero, 7            # encoding: [0xf4,0x01,0xc0,0x00]
# CHECK-TRAP: div $zero, $5, $6           # encoding: [0x1a,0x00,0xa6,0x00]
# CHECK-TRAP: addiu $1, $zero, -1         # encoding: [0xff,0xff,0x01,0x24]
# CHECK-TRAP: bne $6, $1, $tmp1           # encoding: [A,A,0xc1,0x14]
# CHECK-TRAP: lui $1, 32768               # encoding: [0x00,0x80,0x01,0x3c]
# CHECK-TRAP: teq $5, $1, 6               # encoding: [0xb4,0x01,0xa1,0x00]
# CHECK-TRAP: $tmp1
# CHECK-TRAP: mfhi $4                     # encoding: [0x10,0x20,0x00,0x00]

  rem $4,$5,$0
# CHECK-NOTRAP: break 7                   # encoding: [0x0d,0x00,0x07,0x00]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0xf4,0x01,0x00,0x00]

  rem $4,$0,$0
# CHECK-NOTRAP: break 7                   # encoding: [0x0d,0x00,0x07,0x00]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0xf4,0x01,0x00,0x00]

  rem $0,$5,$4
# CHECK-NOTRAP: div $zero, $5, $4         # encoding: [0x1a,0x00,0xa4,0x00]
# CHECK-TRAP: div $zero, $5, $4           # encoding: [0x1a,0x00,0xa4,0x00]

  rem $4,$5,0
# CHECK-NOTRAP: break 7                   # encoding: [0x0d,0x00,0x07,0x00]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0xf4,0x01,0x00,0x00]

  rem $4,$0,0
# CHECK-NOTRAP: break 7                   # encoding: [0x0d,0x00,0x07,0x00]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0xf4,0x01,0x00,0x00]

  rem $4,$5,1
# CHECK-NOTRAP: move $4, $zero            # encoding: [0x25,0x20,0x00,0x00]
# CHECK-TRAP: move $4, $zero              # encoding: [0x25,0x20,0x00,0x00]

  rem $4,$5,-1
# CHECK-NOTRAP: move $4, $zero            # encoding: [0x25,0x20,0x00,0x00]
# CHECK-TRAP: move $4, $zero              # encoding: [0x25,0x20,0x00,0x00]

  rem $4,$5,2
# CHECK-NOTRAP: addiu $1, $zero, 2        # encoding: [0x02,0x00,0x01,0x24]
# CHECK-NOTRAP: div $zero, $5, $1         # encoding: [0x1a,0x00,0xa1,0x00]
# CHECK-NOTRAP: mfhi $4                   # encoding: [0x10,0x20,0x00,0x00]
# CHECK-TRAP: addiu $1, $zero, 2          # encoding: [0x02,0x00,0x01,0x24]
# CHECK-TRAP: div $zero, $5, $1           # encoding: [0x1a,0x00,0xa1,0x00]
# CHECK-TRAP: mfhi $4                     # encoding: [0x10,0x20,0x00,0x00]

  rem $4,$5,0x8000
# CHECK-NOTRAP: ori $1, $zero, 32768      # encoding: [0x00,0x80,0x01,0x34]
# CHECK-NOTRAP: div $zero, $5, $1         # encoding: [0x1a,0x00,0xa1,0x00]
# CHECK-NOTRAP: mfhi $4                   # encoding: [0x10,0x20,0x00,0x00]
# CHECK-TRAP: ori $1, $zero, 32768        # encoding: [0x00,0x80,0x01,0x34]
# CHECK-TRAP: div $zero, $5, $1           # encoding: [0x1a,0x00,0xa1,0x00]
# CHECK-TRAP: mfhi $4                     # encoding: [0x10,0x20,0x00,0x00]


  rem $4,$5,-0x8000
# CHECK-NOTRAP: addiu $1, $zero, -32768   # encoding: [0x00,0x80,0x01,0x24]
# CHECK-NOTRAP: div $zero, $5, $1         # encoding: [0x1a,0x00,0xa1,0x00]
# CHECK-NOTRAP: mfhi $4                   # encoding: [0x10,0x20,0x00,0x00]
# CHECK-TRAP: addiu $1, $zero, -32768     # encoding: [0x00,0x80,0x01,0x24]
# CHECK-TRAP: div $zero, $5, $1           # encoding: [0x1a,0x00,0xa1,0x00]
# CHECK-TRAP: mfhi $4                     # encoding: [0x10,0x20,0x00,0x00]


  rem $4,$5,0x10000
# CHECK-NOTRAP: lui $1, 1                 # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-NOTRAP: div $zero, $5, $1         # encoding: [0x1a,0x00,0xa1,0x00]
# CHECK-NOTRAP: mfhi $4                   # encoding: [0x10,0x20,0x00,0x00]
# CHECK-TRAP: lui $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-TRAP: div $zero, $5, $1           # encoding: [0x1a,0x00,0xa1,0x00]
# CHECK-TRAP: mfhi $4                     # encoding: [0x10,0x20,0x00,0x00]


  rem $4,$5,0x1a5a5
# CHECK-NOTRAP: lui $1, 1                 # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-NOTRAP: ori $1, $1, 42405         # encoding: [0xa5,0xa5,0x21,0x34]
# CHECK-NOTRAP: div $zero, $5, $1         # encoding: [0x1a,0x00,0xa1,0x00]
# CHECK-NOTRAP: mfhi $4                   # encoding: [0x10,0x20,0x00,0x00]
# CHECK-TRAP: lui $1, 1                   # encoding: [0x01,0x00,0x01,0x3c]
# CHECK-TRAP: ori $1, $1, 42405           # encoding: [0xa5,0xa5,0x21,0x34]
# CHECK-TRAP: div $zero, $5, $1           # encoding: [0x1a,0x00,0xa1,0x00]
# CHECK-TRAP: mfhi $4                     # encoding: [0x10,0x20,0x00,0x00]
