# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r2 | \
# RUN:   FileCheck %s --check-prefix=CHECK-NOTRAP
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32r2 \
# RUN:  -mattr=+use-tcc-in-div | FileCheck %s --check-prefix=CHECK-TRAP

  divu $25,$11
# CHECK-NOTRAP: bnez $11, $tmp0           # encoding: [0x15,0x60,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: ($tmp0)-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: divu $zero, $25, $11      # encoding: [0x03,0x2b,0x00,0x1b]
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: $tmp0:
# CHECK-NOTRAP: mflo $25                  # encoding: [0x00,0x00,0xc8,0x12]

  divu $24,$12
# CHECK-NOTRAP: bnez $12, $tmp1           # encoding: [0x15,0x80,A,A]
# CHECK-NOTRAP:                           # fixup A - offset: 0, value: ($tmp1)-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: divu $zero, $24, $12      # encoding: [0x03,0x0c,0x00,0x1b]
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: $tmp1:
# CHECK-NOTRAP: mflo $24                  # encoding: [0x00,0x00,0xc0,0x12]

  divu $25,$0
# CHECK-NOTRAP: break 7                   # encoding: [0x00,0x07,0x00,0x0d]

  divu $0,$9
# CHECK-NOTRAP: divu $zero, $zero, $9     # encoding: [0x00,0x09,0x00,0x1b]

  divu $0,$0
# CHECK-NOTRAP: divu $zero, $zero, $zero  # encoding: [0x00,0x00,0x00,0x1b]

   divu $4,$5,$6
# CHECK-NOTRAP: bnez $6, $tmp2             # encoding: [0x14,0xc0,A,A]
# CHECK-NOTRAP:                            # fixup A - offset: 0, value: ($tmp2)-4, kind: fixup_Mips_PC16
# CHECK-NOTRAP: divu $zero, $5, $6         # encoding: [0x00,0xa6,0x00,0x1b]
# CHECK-NOTRAP: break 7                    # encoding: [0x00,0x07,0x00,0x0d]
# CHECK-NOTRAP: $tmp2:
# CHECK-NOTRAP: mflo $4                    # encoding: [0x00,0x00,0x20,0x12]

   divu $4,$5,$0
# CHECK-NOTRAP: break 7                    # encoding: [0x00,0x07,0x00,0x0d]

   divu $4,$0,$0
# CHECK-NOTRAP: break 7                    # encoding: [0x00,0x07,0x00,0x0d]

  divu $0, $4, $5
# CHECK-NOTRAP: divu $zero, $4, $5         # encoding: [0x00,0x85,0x00,0x1b]

   divu $25, $11
# CHECK-TRAP: teq $11, $zero, 7            # encoding: [0x01,0x60,0x01,0xf4]
# CHECK-TRAP: divu $zero, $25, $11         # encoding: [0x03,0x2b,0x00,0x1b]
# CHECK-TRAP: mflo $25                     # encoding: [0x00,0x00,0xc8,0x12]

   divu $24,$12
# CHECK-TRAP: teq $12, $zero, 7            # encoding: [0x01,0x80,0x01,0xf4]
# CHECK-TRAP: divu $zero, $24, $12         # encoding: [0x03,0x0c,0x00,0x1b]
# CHECK-TRAP: mflo $24                     # encoding: [0x00,0x00,0xc0,0x12]

  divu $25,$0
# CHECK-TRAP: teq $zero, $zero, 7          # encoding: [0x00,0x00,0x01,0xf4]

  divu $0,$9
# CHECK-TRAP: divu $zero, $zero, $9        # encoding: [0x00,0x09,0x00,0x1b]

  divu $0,$0
# CHECK-TRAP: divu $zero, $zero, $zero     # encoding: [0x00,0x00,0x00,0x1b]

  divu $4,$5,$6
# CHECK-TRAP: teq $6, $zero, 7             # encoding: [0x00,0xc0,0x01,0xf4]
# CHECK-TRAP: divu $zero, $5, $6           # encoding: [0x00,0xa6,0x00,0x1b]
# CHECK-TRAP: mflo $4                      # encoding: [0x00,0x00,0x20,0x12]

  divu $4,$5,$0
# CHECK-TRAP: teq $zero, $zero, 7          # encoding: [0x00,0x00,0x01,0xf4]

  divu $4,$0,$0
# CHECK-TRAP: teq $zero, $zero, 7          # encoding: [0x00,0x00,0x01,0xf4]
# CHECK-TRAP: divu $zero, $zero, $zero     # encoding: [0x00,0x00,0x00,0x1b]
# CHECK-TRAP: mflo $4                      # encoding: [0x00,0x00,0x20,0x12]

  divu $0, $4, $5
# CHECK-TRAP: divu $zero, $4, $5           # encoding: [0x00,0x85,0x00,0x1b]
