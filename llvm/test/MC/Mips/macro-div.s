# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r2 | \
# RUN:   FileCheck %s --check-prefix=CHECK-NOTRAP
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r2 \
# RUN:  -mattr=+use-tcc-in-div | FileCheck %s --check-prefix=CHECK-TRAP

  div $25,$11
# CHECK-NOTRAP: bnez $11, $tmp0           # encoding: [0x15,0x60,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: ($tmp0)-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: div $zero, $25, $11       # encoding: [0x03,0x2b,0x00,0x1a]
# CHECK-NOTRAP: break  7                  # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: $tmp0:
# CHECK-NOTRAP: addiu $1, $zero, -1       # encoding: [0x24,0x01,0xff,0xff]
# CHECK-NOTRAP: bne $11, $1, $tmp1        # encoding: [0x15,0x61,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: ($tmp1)-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: lui $1, 32768             # encoding: [0x3c,0x01,0x80,0x00]
# CHECK-NOTRAP: bne $25, $1, $tmp1        # encoding: [0x17,0x21,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: ($tmp1)-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: nop                       # encoding: [0x00,0x00,0x00,0x00]
# CHECK-NOTRAP: break 6                   # encoding: [0x00,0x06,0x00,0x0d]
# CHECK-NOTRAP: $tmp1:
# CHECK-NOTRAP: mflo $25                  # encoding: [0x00,0x00,0xc8,0x12]
# CHECK-TRAP: teq $11, $zero, 7           # encoding: [0x01,0x60,0x01,0xf4]
# CHECK-TRAP: div $zero, $25, $11         # encoding: [0x03,0x2b,0x00,0x1a]
# CHECK-TRAP: addiu $1, $zero, -1         # encoding: [0x24,0x01,0xff,0xff]
# CHECK-TRAP: bne $11, $1, $tmp0          # encoding: [0x15,0x61,A,A]
# CHECK-TRAP:                             # fixup A - offset: 0, value: ($tmp0)-4, kind: fixup_Mips_PC16
# CHECK-TRAP: lui $1, 32768               # encoding: [0x3c,0x01,0x80,0x00]
# CHECK-TRAP: teq $25, $1, 6              # encoding: [0x03,0x21,0x01,0xb4]
# CHECK-TRAP: $tmp0:
# CHECK-TRAP: mflo $25                    # encoding: [0x00,0x00,0xc8,0x12]

  div $24,$12
# CHECK-NOTRAP: bnez $12, $tmp2           # encoding: [0x15,0x80,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: ($tmp2)-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: div $zero, $24, $12       # encoding: [0x03,0x0c,0x00,0x1a]
# CHECK-NOTRAP: break  7                  # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: $tmp2:
# CHECK-NOTRAP: addiu $1, $zero, -1       # encoding: [0x24,0x01,0xff,0xff]
# CHECK-NOTRAP: bne $12, $1, $tmp3        # encoding: [0x15,0x81,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: ($tmp3)-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: lui $1, 32768             # encoding: [0x3c,0x01,0x80,0x00]
# CHECK-NOTRAP: bne $24, $1, $tmp3        # encoding: [0x17,0x01,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: ($tmp3)-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: nop                       # encoding: [0x00,0x00,0x00,0x00]
# CHECK-NOTRAP: break  6                  # encoding: [0x00,0x06,0x00,0x0d]
# CHECK-NOTRAP: $tmp3:
# CHECK-NOTRAP: mflo $24                  # encoding: [0x00,0x00,0xc0,0x12]
# CHECK-TRAP: teq $12, $zero, 7           # encoding: [0x01,0x80,0x01,0xf4]
# CHECK-TRAP: div $zero, $24, $12         # encoding: [0x03,0x0c,0x00,0x1a]
# CHECK-TRAP: addiu $1, $zero, -1         # encoding: [0x24,0x01,0xff,0xff]
# CHECK-TRAP: bne $12, $1, $tmp1          # encoding: [0x15,0x81,A,A]
# CHECK-TRAP:                             # fixup A - offset: 0, value: ($tmp1)-4, kind: fixup_Mips_PC16
# CHECK-TRAP: lui $1, 32768               # encoding: [0x3c,0x01,0x80,0x00]
# CHECK-TRAP: teq $24, $1, 6              # encoding: [0x03,0x01,0x01,0xb4]
# CHECK-TRAP: $tmp1:
# CHECK-TRAP: mflo $24                    # encoding: [0x00,0x00,0xc0,0x12]

  div $25,$0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  div $0,$9
# CHECK-NOTRAP: div $zero, $zero, $9      # encoding: [0x00,0x09,0x00,0x1a]
# CHECK-TRAP: div $zero, $zero, $9        # encoding: [0x00,0x09,0x00,0x1a]

  div $0,$0
# CHECK-NOTRAP: div $zero, $zero, $zero   # encoding: [0x00,0x00,0x00,0x1a]
# CHECK-TRAP: div $zero, $zero, $zero     # encoding: [0x00,0x00,0x00,0x1a]

  div $4,0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  div $0,0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  div $4,1
# CHECK-NOTRAP: move $4, $4               # encoding: [0x00,0x80,0x20,0x25]
# CHECK-TRAP: move $4, $4                 # encoding: [0x00,0x80,0x20,0x25]

  div $4,-1
# CHECK-NOTRAP: neg  $4, $4               # encoding: [0x00,0x04,0x20,0x22]
# CHECK-TRAP: neg  $4, $4                 # encoding: [0x00,0x04,0x20,0x22]

  div $4,2
# CHECK-NOTRAP: addiu $1, $zero, 2        # encoding: [0x24,0x01,0x00,0x02]
# CHECK-NOTRAP: div $zero, $4, $1         # encoding: [0x00,0x81,0x00,0x1a]
# CHECK-NOTRAP: mflo  $4                  # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: addiu $1, $zero, 2          # encoding: [0x24,0x01,0x00,0x02]
# CHECK-TRAP: div $zero, $4, $1           # encoding: [0x00,0x81,0x00,0x1a]
# CHECK-TRAP: mflo  $4                    # encoding: [0x00,0x00,0x20,0x12]

  div $4,0x8000
# CHECK-NOTRAP: ori $1, $zero, 32768      # encoding: [0x34,0x01,0x80,0x00]
# CHECK-NOTRAP: div $zero, $4, $1         # encoding: [0x00,0x81,0x00,0x1a]
# CHECK-NOTRAP: mflo  $4                  # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: ori $1, $zero, 32768        # encoding: [0x34,0x01,0x80,0x00]
# CHECK-TRAP: div $zero, $4, $1           # encoding: [0x00,0x81,0x00,0x1a]
# CHECK-TRAP: mflo  $4                    # encoding: [0x00,0x00,0x20,0x12]

  div $4,-0x8000
# CHECK-NOTRAP: addiu $1, $zero, -32768   # encoding: [0x24,0x01,0x80,0x00]
# CHECK-NOTRAP: div $zero, $4, $1         # encoding: [0x00,0x81,0x00,0x1a]
# CHECK-NOTRAP: mflo  $4                  # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: addiu $1, $zero, -32768     # encoding: [0x24,0x01,0x80,0x00]
# CHECK-TRAP: div $zero, $4, $1           # encoding: [0x00,0x81,0x00,0x1a]
# CHECK-TRAP: mflo  $4                    # encoding: [0x00,0x00,0x20,0x12]

  div $4,0x10000
# CHECK-NOTRAP: lui $1, 1                 # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-NOTRAP: div $zero, $4, $1         # encoding: [0x00,0x81,0x00,0x1a]
# CHECK-NOTRAP: mflo  $4                  # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: lui $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-TRAP: div $zero, $4, $1           # encoding: [0x00,0x81,0x00,0x1a]
# CHECK-TRAP: mflo  $4                    # encoding: [0x00,0x00,0x20,0x12]

  div $4,0x1a5a5
# CHECK-NOTRAP: lui $1, 1                 # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-NOTRAP: ori $1, $1, 42405         # encoding: [0x34,0x21,0xa5,0xa5]
# CHECK-NOTRAP: div $zero, $4, $1         # encoding: [0x00,0x81,0x00,0x1a]
# CHECK-NOTRAP: mflo  $4                  # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: lui $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-TRAP: ori $1, $1, 42405           # encoding: [0x34,0x21,0xa5,0xa5]
# CHECK-TRAP: div $zero, $4, $1           # encoding: [0x00,0x81,0x00,0x1a]
# CHECK-TRAP: mflo  $4                    # encoding: [0x00,0x00,0x20,0x12]

  div $4,$5,$6
# CHECK-NOTRAP: bnez $6, $tmp4            # encoding: [0x14,0xc0,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: ($tmp4)-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: div $zero, $5, $6         # encoding: [0x00,0xa6,0x00,0x1a]
# CHECK-NOTRAP: break  7                  # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: $tmp4:
# CHECK-NOTRAP: addiu $1, $zero, -1       # encoding: [0x24,0x01,0xff,0xff]
# CHECK-NOTRAP: bne $6, $1, $tmp5         # encoding: [0x14,0xc1,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: ($tmp5)-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: lui $1, 32768             # encoding: [0x3c,0x01,0x80,0x00]
# CHECK-NOTRAP: bne $5, $1, $tmp5         # encoding: [0x14,0xa1,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: ($tmp5)-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: nop                       # encoding: [0x00,0x00,0x00,0x00]
# CHECK-NOTRAP: break 6                   # encoding: [0x00,0x06,0x00,0x0d]
# CHECK-NOTRAP: $tmp5:
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: teq $6, $zero, 7            # encoding: [0x00,0xc0,0x01,0xf4]
# CHECK-TRAP: div $zero, $5, $6           # encoding: [0x00,0xa6,0x00,0x1a]
# CHECK-TRAP: addiu $1, $zero, -1         # encoding: [0x24,0x01,0xff,0xff]
# CHECK-TRAP: bne $6, $1, $tmp2           # encoding: [0x14,0xc1,A,A]
# CHECK-TRAP:                             # fixup A - offset: 0, value: ($tmp2)-4, kind: fixup_Mips_PC16
# CHECK-TRAP: lui $1, 32768               # encoding: [0x3c,0x01,0x80,0x00]
# CHECK-TRAP: teq $5, $1, 6               # encoding: [0x00,0xa1,0x01,0xb4]
# CHECK-TRAP: $tmp2:
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  div $4,$5,$0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  div $4,$0,$0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  div $0,$4,$5
# CHECK-NOTRAP: div $zero, $4, $5         # encoding: [0x00,0x85,0x00,0x1a]
# CHECK-TRAP: div $zero, $4, $5           # encoding: [0x00,0x85,0x00,0x1a]

  div $4,$5,0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  div $4,$0,0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  div $4,$5,1
# CHECK-NOTRAP: move $4, $5               # encoding: [0x00,0xa0,0x20,0x25]
# CHECK-TRAP: move $4, $5                 # encoding: [0x00,0xa0,0x20,0x25]

  div $4,$5,-1
# CHECK-NOTRAP: neg  $4, $5               # encoding: [0x00,0x05,0x20,0x22]
# CHECK-TRAP: neg  $4, $5                 # encoding: [0x00,0x05,0x20,0x22]

  div $4,$5,2
# CHECK-NOTRAP: addiu $1, $zero, 2        # encoding: [0x24,0x01,0x00,0x02]
# CHECK-NOTRAP: div $zero, $5, $1         # encoding: [0x00,0xa1,0x00,0x1a]
# CHECK-NOTRAP: mflo  $4                  # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: addiu $1, $zero, 2          # encoding: [0x24,0x01,0x00,0x02]
# CHECK-TRAP: div $zero, $5, $1           # encoding: [0x00,0xa1,0x00,0x1a]
# CHECK-TRAP: mflo  $4                    # encoding: [0x00,0x00,0x20,0x12]

  div $4,$5,0x8000
# CHECK-NOTRAP: ori $1, $zero, 32768      # encoding: [0x34,0x01,0x80,0x00]
# CHECK-NOTRAP: div $zero, $5, $1         # encoding: [0x00,0xa1,0x00,0x1a]
# CHECK-NOTRAP: mflo  $4                  # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: ori $1, $zero, 32768        # encoding: [0x34,0x01,0x80,0x00]
# CHECK-TRAP: div $zero, $5, $1           # encoding: [0x00,0xa1,0x00,0x1a]
# CHECK-TRAP: mflo  $4                    # encoding: [0x00,0x00,0x20,0x12]

  div $4,$5,-0x8000
# CHECK-NOTRAP: addiu $1, $zero, -32768   # encoding: [0x24,0x01,0x80,0x00]
# CHECK-NOTRAP: div $zero, $5, $1         # encoding: [0x00,0xa1,0x00,0x1a]
# CHECK-NOTRAP: mflo  $4                  # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: addiu $1, $zero, -32768     # encoding: [0x24,0x01,0x80,0x00]
# CHECK-TRAP: div $zero, $5, $1           # encoding: [0x00,0xa1,0x00,0x1a]
# CHECK-TRAP: mflo  $4                    # encoding: [0x00,0x00,0x20,0x12]

  div $4,$5,0x10000
# CHECK-NOTRAP: lui $1, 1                 # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-NOTRAP: div $zero, $5, $1         # encoding: [0x00,0xa1,0x00,0x1a]
# CHECK-NOTRAP: mflo  $4                  # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: lui $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-TRAP: div $zero, $5, $1           # encoding: [0x00,0xa1,0x00,0x1a]
# CHECK-TRAP: mflo  $4                    # encoding: [0x00,0x00,0x20,0x12]

  div $4,$5,0x1a5a5
# CHECK-NOTRAP: lui $1, 1                 # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-NOTRAP: ori $1, $1, 42405         # encoding: [0x34,0x21,0xa5,0xa5]
# CHECK-NOTRAP: div $zero, $5, $1         # encoding: [0x00,0xa1,0x00,0x1a]
# CHECK-NOTRAP: mflo  $4                  # encoding: [0x00,0x00,0x20,0x12]
# CHECK-TRAP: lui $1, 1                   # encoding: [0x3c,0x01,0x00,0x01]
# CHECK-TRAP: ori $1, $1, 42405           # encoding: [0x34,0x21,0xa5,0xa5]
# CHECK-TRAP: div $zero, $5, $1           # encoding: [0x00,0xa1,0x00,0x1a]
# CHECK-TRAP: mflo  $4                    # encoding: [0x00,0x00,0x20,0x12]
