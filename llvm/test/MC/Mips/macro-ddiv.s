# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips64r2 | \
# RUN:   FileCheck %s --check-prefix=CHECK-NOTRAP
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips64r2 \
# RUN:  -mattr=+use-tcc-in-div | FileCheck %s --check-prefix=CHECK-TRAP

  ddiv $25, $11
# CHECK-NOTRAP: bne $11, $zero, 8         # encoding: [0x15,0x60,0x00,0x02]
# CHECK-NOTRAP: ddiv $zero, $25, $11      # encoding: [0x03,0x2b,0x00,0x1e]
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: addiu $1, $zero, -1       # encoding: [0x24,0x01,0xff,0xff]
# CHECK-NOTRAP: bne $11, $1, 20           # encoding: [0x15,0x61,0x00,0x05]
# CHECK-NOTRAP: addiu $1, $zero, 1        # encoding: [0x24,0x01,0x00,0x01]
# CHECK-NOTRAP: dsll32 $1, $1, 31         # encoding: [0x00,0x01,0x0f,0xfc]
# CHECK-NOTRAP: bne $25, $1, 8            # encoding: [0x17,0x21,0x00,0x02]
# CHECK-NOTRAP: sll $zero, $zero, 0       # encoding: [0x00,0x00,0x00,0x00]
# CHECK-NOTRAP: break 6                   # encoding: [0x00,0x06,0x00,0x0d]
# CHECK-NOTRAP: mflo $25                  # encoding: [0x00,0x00,0xc8,0x12]

  ddiv $24,$12
# CHECK-NOTRAP: bne $12, $zero, 8         # encoding: [0x15,0x80,0x00,0x02]
# CHECK-NOTRAP: ddiv $zero, $24, $12      # encoding: [0x03,0x0c,0x00,0x1e]
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: addiu $1, $zero, -1       # encoding: [0x24,0x01,0xff,0xff]
# CHECK-NOTRAP: bne $12, $1, 20           # encoding: [0x15,0x81,0x00,0x05]
# CHECK-NOTRAP: addiu $1, $zero, 1        # encoding: [0x24,0x01,0x00,0x01]
# CHECK-NOTRAP: dsll32 $1, $1, 31         # encoding: [0x00,0x01,0x0f,0xfc]
# CHECK-NOTRAP: bne $24, $1, 8            # encoding: [0x17,0x01,0x00,0x02]
# CHECK-NOTRAP: sll $zero, $zero, 0       # encoding: [0x00,0x00,0x00,0x00]
# CHECK-NOTRAP: break 6                   # encoding: [0x00,0x06,0x00,0x0d]
# CHECK-NOTRAP: mflo $24                  # encoding: [0x00,0x00,0xc0,0x12]

  ddiv $25,$0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]

  ddiv $0,$9
# CHECK-NOTRAP: bne $9, $zero, 8          # encoding: [0x15,0x20,0x00,0x02]
# CHECK-NOTRAP: ddiv $zero, $zero, $9     # encoding: [0x00,0x09,0x00,0x1e]
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: addiu $1, $zero, -1       # encoding: [0x24,0x01,0xff,0xff]
# CHECK-NOTRAP: bne $9, $1, 20            # encoding: [0x15,0x21,0x00,0x05]
# CHECK-NOTRAP: addiu $1, $zero, 1        # encoding: [0x24,0x01,0x00,0x01]
# CHECK-NOTRAP: dsll32 $1, $1, 31         # encoding: [0x00,0x01,0x0f,0xfc]
# CHECK-NOTRAP: bne $zero, $1, 8          # encoding: [0x14,0x01,0x00,0x02]
# CHECK-NOTRAP: sll $zero, $zero, 0       # encoding: [0x00,0x00,0x00,0x00]
# CHECK-NOTRAP: break 6                   # encoding: [0x00,0x06,0x00,0x0d]
# CHECK-NOTRAP: mflo $zero                # encoding: [0x00,0x00,0x00,0x12]

  ddiv $0,$0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]

  ddiv  $4,$5,$6
# CHECK-NOTRAP: bne $6, $zero, 8          # encoding: [0x14,0xc0,0x00,0x02]
# CHECK-NOTRAP: ddiv $zero, $5, $6        # encoding: [0x00,0xa6,0x00,0x1e]
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: addiu $1, $zero, -1       # encoding: [0x24,0x01,0xff,0xff]
# CHECK-NOTRAP: bne $6, $1, 20            # encoding: [0x14,0xc1,0x00,0x05]
# CHECK-NOTRAP: addiu $1, $zero, 1        # encoding: [0x24,0x01,0x00,0x01]
# CHECK-NOTRAP: dsll32 $1, $1, 31         # encoding: [0x00,0x01,0x0f,0xfc]
# CHECK-NOTRAP: bne $5, $1, 8             # encoding: [0x14,0xa1,0x00,0x02]
# CHECK-NOTRAP: sll $zero, $zero, 0       # encoding: [0x00,0x00,0x00,0x00]
# CHECK-NOTRAP: break 6                   # encoding: [0x00,0x06,0x00,0x0d]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]

  ddiv  $4,$5,$0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]

  ddiv  $4,$0,$0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]

  ddiv $0, $4, $5
# CHECK-NOTRAP: ddiv $zero, $4, $5        # encoding: [0x00,0x85,0x00,0x1e]

  ddiv $25,$11
# CHECK-TRAP: teq $11, $zero, 7           # encoding: [0x01,0x60,0x01,0xf4]
# CHECK-TRAP: ddiv $zero, $25, $11        # encoding: [0x03,0x2b,0x00,0x1e]
# CHECK-TRAP: addiu $1, $zero, -1         # encoding: [0x24,0x01,0xff,0xff]
# CHECK-TRAP: bne $11, $1, 12             # encoding: [0x15,0x61,0x00,0x03]
# CHECK-TRAP: addiu $1, $zero, 1          # encoding: [0x24,0x01,0x00,0x01]
# CHECK-TRAP: dsll32 $1, $1, 31           # encoding: [0x00,0x01,0x0f,0xfc]
# CHECK-TRAP: teq $25, $1, 6              # encoding: [0x03,0x21,0x01,0xb4]
# CHECK-TRAP: mflo $25                    # encoding: [0x00,0x00,0xc8,0x12]

  ddiv $24,$12
# CHECK-TRAP: teq $12, $zero, 7           # encoding: [0x01,0x80,0x01,0xf4]
# CHECK-TRAP: ddiv $zero, $24, $12        # encoding: [0x03,0x0c,0x00,0x1e]
# CHECK-TRAP: addiu $1, $zero, -1         # encoding: [0x24,0x01,0xff,0xff]
# CHECK-TRAP: bne $12, $1, 12             # encoding: [0x15,0x81,0x00,0x03]
# CHECK-TRAP: addiu $1, $zero, 1          # encoding: [0x24,0x01,0x00,0x01]
# CHECK-TRAP: dsll32 $1, $1, 31           # encoding: [0x00,0x01,0x0f,0xfc]
# CHECK-TRAP: teq $24, $1, 6              # encoding: [0x03,0x01,0x01,0xb4]
# CHECK-TRAP: mflo $24                    # encoding: [0x00,0x00,0xc0,0x12]

  ddiv $25,$0
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  ddiv $0,$9
# CHECK-TRAP: teq $9, $zero, 7            # encoding: [0x01,0x20,0x01,0xf4]
# CHECK-TRAP: ddiv $zero, $zero, $9       # encoding: [0x00,0x09,0x00,0x1e]
# CHECK-TRAP: addiu $1, $zero, -1         # encoding: [0x24,0x01,0xff,0xff]
# CHECK-TRAP: bne $9, $1, 12              # encoding: [0x15,0x21,0x00,0x03]
# CHECK-TRAP: addiu $1, $zero, 1          # encoding: [0x24,0x01,0x00,0x01]
# CHECK-TRAP: dsll32 $1, $1, 31           # encoding: [0x00,0x01,0x0f,0xfc]
# CHECK-TRAP: teq $zero, $1, 6            # encoding: [0x00,0x01,0x01,0xb4]
# CHECK-TRAP: mflo $zero                  # encoding: [0x00,0x00,0x00,0x12]

  ddiv $0,$0
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  ddiv  $4,$5,$6
# CHECK-TRAP: teq $6, $zero, 7            # encoding: [0x00,0xc0,0x01,0xf4]
# CHECK-TRAP: ddiv $zero, $5, $6          # encoding: [0x00,0xa6,0x00,0x1e]
# CHECK-TRAP: addiu $1, $zero, -1         # encoding: [0x24,0x01,0xff,0xff]
# CHECK-TRAP: bne $6, $1, 12              # encoding: [0x14,0xc1,0x00,0x03]
# CHECK-TRAP: addiu $1, $zero, 1          # encoding: [0x24,0x01,0x00,0x01]
# CHECK-TRAP: dsll32 $1, $1, 31           # encoding: [0x00,0x01,0x0f,0xfc]
# CHECK-TRAP: teq $5, $1, 6               # encoding: [0x00,0xa1,0x01,0xb4]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddiv  $4,$5,$0
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  ddiv  $4,$0,$0
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]

  ddiv $0, $4, $5
# CHECK-TRAP: ddiv $zero, $4, $5          # encoding: [0x00,0x85,0x00,0x1e]
