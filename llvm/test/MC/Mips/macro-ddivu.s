# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips64r2 | \
# RUN:   FileCheck %s --check-prefix=CHECK-NOTRAP
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips64r2 \
# RUN:  -mattr=+use-tcc-in-div | FileCheck %s --check-prefix=CHECK-TRAP

  ddivu $25,$11
# CHECK-NOTRAP: bne $11, $zero, 8         # encoding: [0x15,0x60,0x00,0x02]
# CHECK-NOTRAP: ddivu $zero, $25, $11     # encoding: [0x03,0x2b,0x00,0x1f]
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: mflo $25                  # encoding: [0x00,0x00,0xc8,0x12]

  ddivu $24,$12
# CHECK-NOTRAP: bne $12, $zero, 8         # encoding: [0x15,0x80,0x00,0x02]
# CHECK-NOTRAP: ddivu $zero, $24, $12     # encoding: [0x03,0x0c,0x00,0x1f]
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: mflo $24                  # encoding: [0x00,0x00,0xc0,0x12]

  ddivu $25,$0
# CHECK-NOTRAP: bne $zero, $zero, 8       # encoding: [0x14,0x00,0x00,0x02]
# CHECK-NOTRAP: ddivu $zero, $25, $zero   # encoding: [0x03,0x20,0x00,0x1f]
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: mflo $25                  # encoding: [0x00,0x00,0xc8,0x12]

  ddivu $0,$9
# CHECK-NOTRAP: bne $9, $zero, 8          # encoding: [0x15,0x20,0x00,0x02]
# CHECK-NOTRAP: ddivu $zero, $zero, $9    # encoding: [0x00,0x09,0x00,0x1f]
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: mflo $zero                # encoding: [0x00,0x00,0x00,0x12]

  ddivu $0,$0
# CHECK-NOTRAP: bne $zero, $zero, 8       # encoding: [0x14,0x00,0x00,0x02]
# CHECK-NOTRAP: ddivu $zero, $zero, $zero # encoding: [0x00,0x00,0x00,0x1f]
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: mflo $zero                # encoding: [0x00,0x00,0x00,0x12]

  ddivu  $4,$5,$6
# CHECK-NOTRAP: bne $6, $zero, 8          # encoding: [0x14,0xc0,0x00,0x02]
# CHECK-NOTRAP: ddivu $zero, $5, $6       # encoding: [0x00,0xa6,0x00,0x1f]
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]

  ddivu $4,$5,$0
# CHECK-NOTRAP: bne $zero, $zero, 8       # encoding: [0x14,0x00,0x00,0x02]
# CHECK-NOTRAP: ddivu $zero, $5, $zero    # encoding: [0x00,0xa0,0x00,0x1f]
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]

  ddivu $4,$0,$0
# CHECK-NOTRAP: bne $zero, $zero, 8       # encoding: [0x14,0x00,0x00,0x02]
# CHECK-NOTRAP: ddivu $zero, $zero, $zero # encoding: [0x00,0x00,0x00,0x1f]
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: mflo $4                   # encoding: [0x00,0x00,0x20,0x12]

  ddivu $0, $4, $5
# CHECK-NOTRAP: ddivu $zero, $4, $5       # encoding: [0x00,0x85,0x00,0x1f]

  ddivu $25, $11
# CHECK-TRAP: teq $11, $zero, 7           # encoding: [0x01,0x60,0x01,0xf4]
# CHECK-TRAP: ddivu $zero, $25, $11       # encoding: [0x03,0x2b,0x00,0x1f]
# CHECK-TRAP: mflo $25                    # encoding: [0x00,0x00,0xc8,0x12]

  ddivu $24,$12
# CHECK-TRAP: teq $12, $zero, 7           # encoding: [0x01,0x80,0x01,0xf4]
# CHECK-TRAP: ddivu $zero, $24, $12       # encoding: [0x03,0x0c,0x00,0x1f]
# CHECK-TRAP: mflo $24                    # encoding: [0x00,0x00,0xc0,0x12]

  ddivu $25,$0
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]
# CHECK-TRAP: ddivu $zero, $25, $zero     # encoding: [0x03,0x20,0x00,0x1f]
# CHECK-TRAP: mflo $25                    # encoding: [0x00,0x00,0xc8,0x12]

  ddivu $0,$9
# CHECK-TRAP: teq $9, $zero, 7            # encoding: [0x01,0x20,0x01,0xf4]
# CHECK-TRAP: ddivu $zero, $zero, $9      # encoding: [0x00,0x09,0x00,0x1f]
# CHECK-TRAP: mflo $zero                  # encoding: [0x00,0x00,0x00,0x12]

  ddivu $0,$0
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]
# CHECK-TRAP: ddivu $zero, $zero, $zero   # encoding: [0x00,0x00,0x00,0x1f]
# CHECK-TRAP: mflo $zero                  # encoding: [0x00,0x00,0x00,0x12]

  ddivu  $4,$5,$6
# CHECK-TRAP: teq $6, $zero, 7            # encoding: [0x00,0xc0,0x01,0xf4]
# CHECK-TRAP: ddivu $zero, $5, $6         # encoding: [0x00,0xa6,0x00,0x1f]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddivu $4,$5,$0
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]
# CHECK-TRAP: ddivu $zero, $5, $zero      # encoding: [0x00,0xa0,0x00,0x1f]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddivu $4,$0,$0
# CHECK-TRAP: teq $zero, $zero, 7         # encoding: [0x00,0x00,0x01,0xf4]
# CHECK-TRAP: ddivu $zero, $zero, $zero   # encoding: [0x00,0x00,0x00,0x1f]
# CHECK-TRAP: mflo $4                     # encoding: [0x00,0x00,0x20,0x12]

  ddivu $0, $4, $5
# CHECK-TRAP: ddivu $zero, $4, $5         # encoding: [0x00,0x85,0x00,0x1f]
