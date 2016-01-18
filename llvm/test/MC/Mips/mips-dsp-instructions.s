# RUN: llvm-mc -show-encoding -triple=mips-unknown-unknown -mattr=dspr2 %s | FileCheck %s
#
# CHECK:   .text
# CHECK:   precrq.qb.ph      $16, $17, $18   # encoding: [0x7e,0x32,0x83,0x11]
# CHECK:   precrq.ph.w       $17, $18, $19   # encoding: [0x7e,0x53,0x8d,0x11]
# CHECK:   precrq_rs.ph.w    $18, $19, $20   # encoding: [0x7e,0x74,0x95,0x51]
# CHECK:   precrqu_s.qb.ph   $19, $20, $21   # encoding: [0x7e,0x95,0x9b,0xd1]
# CHECK:   preceq.w.phl      $20, $21        # encoding: [0x7c,0x15,0xa3,0x12]
# CHECK:   preceq.w.phr      $21, $22        # encoding: [0x7c,0x16,0xab,0x52]
# CHECK:   precequ.ph.qbl    $22, $23        # encoding: [0x7c,0x17,0xb1,0x12]
# CHECK:   precequ.ph.qbr    $23, $24        # encoding: [0x7c,0x18,0xb9,0x52]
# CHECK:   precequ.ph.qbla   $24, $25        # encoding: [0x7c,0x19,0xc1,0x92]
# CHECK:   precequ.ph.qbra   $25, $26        # encoding: [0x7c,0x1a,0xc9,0xd2]
# CHECK:   preceu.ph.qbl     $26, $27        # encoding: [0x7c,0x1b,0xd7,0x12]
# CHECK:   preceu.ph.qbr     $27, $gp        # encoding: [0x7c,0x1c,0xdf,0x52]
# CHECK:   preceu.ph.qbla    $gp, $sp        # encoding: [0x7c,0x1d,0xe7,0x92]
# CHECK:   preceu.ph.qbra    $sp, $fp        # encoding: [0x7c,0x1e,0xef,0xd2]

# CHECK:   precr.qb.ph       $23, $24, $25   # encoding: [0x7f,0x19,0xbb,0x51]
# CHECK:   precr_sra.ph.w    $24, $25, 0     # encoding: [0x7f,0x38,0x07,0x91]
# CHECK:   precr_sra.ph.w    $24, $25, 31    # encoding: [0x7f,0x38,0xff,0x91]
# CHECK:   precr_sra_r.ph.w  $25, $26, 0     # encoding: [0x7f,0x59,0x07,0xd1]
# CHECK:   precr_sra_r.ph.w  $25, $26, 31    # encoding: [0x7f,0x59,0xff,0xd1]

# CHECK:   lbux $10, $20($26)                # encoding: [0x7f,0x54,0x51,0x8a]
# CHECK:   lhx  $11, $21($27)                # encoding: [0x7f,0x75,0x59,0x0a]
# CHECK:   lwx  $12, $22($gp)                # encoding: [0x7f,0x96,0x60,0x0a]

# CHECK:    mult $ac3, $2, $3               # encoding: [0x00,0x43,0x18,0x18]
# CHECK:    multu $ac2, $4, $5              # encoding: [0x00,0x85,0x10,0x19]
# CHECK:    madd $ac1, $6, $7               # encoding: [0x70,0xc7,0x08,0x00]
# CHECK:    maddu $ac0, $8, $9              # encoding: [0x71,0x09,0x00,0x01]
# CHECK:    msub $ac3, $10, $11             # encoding: [0x71,0x4b,0x18,0x04]
# CHECK:    msubu $ac2, $12, $13            # encoding: [0x71,0x8d,0x10,0x05]
# CHECK:    mfhi $14, $ac1                  # encoding: [0x00,0x20,0x70,0x10]
# CHECK:    mflo $15, $ac0                  # encoding: [0x00,0x00,0x78,0x12]
# CHECK:    mthi $16, $ac3                  # encoding: [0x02,0x00,0x18,0x11]
# CHECK:    mtlo $17, $ac2                  # encoding: [0x02,0x20,0x10,0x13]

# CHECK:    mult $2, $3                      # encoding: [0x00,0x43,0x00,0x18]
# CHECK:    multu $4, $5                     # encoding: [0x00,0x85,0x00,0x19]
# CHECK:    madd $6, $7                      # encoding: [0x70,0xc7,0x00,0x00]
# CHECK:    maddu $8, $9                     # encoding: [0x71,0x09,0x00,0x01]
# CHECK:    msub $10, $11                    # encoding: [0x71,0x4b,0x00,0x04]
# CHECK:    msubu $12, $13                   # encoding: [0x71,0x8d,0x00,0x05]
# CHECK:    mfhi $14                         # encoding: [0x00,0x00,0x70,0x10]
# CHECK:    mflo $15                         # encoding: [0x00,0x00,0x78,0x12]
# CHECK:    mthi $16                         # encoding: [0x02,0x00,0x00,0x11]
# CHECK:    mtlo $17                         # encoding: [0x02,0x20,0x00,0x13]


  precrq.qb.ph    $16,$17,$18
  precrq.ph.w     $17,$18,$19
  precrq_rs.ph.w  $18,$19,$20
  precrqu_s.qb.ph $19,$20,$21
  preceq.w.phl    $20,$21
  preceq.w.phr    $21,$22
  precequ.ph.qbl  $22,$23
  precequ.ph.qbr  $23,$24
  precequ.ph.qbla $24,$25
  precequ.ph.qbra $25,$26
  preceu.ph.qbl   $26,$27
  preceu.ph.qbr   $27,$28
  preceu.ph.qbla  $28,$29
  preceu.ph.qbra  $29,$30

  precr.qb.ph     $23,$24,$25
  precr_sra.ph.w  $24,$25,0
  precr_sra.ph.w  $24,$25,31
  precr_sra_r.ph.w  $25,$26,0
  precr_sra_r.ph.w  $25,$26,31

  lbux $10, $s4($26)
  lhx  $11, $s5($27)
  lwx  $12, $s6($28)

  mult $ac3, $2, $3
  multu $ac2, $4, $5
  madd $ac1, $6, $7
  maddu $ac0, $8, $9
  msub $ac3, $10, $11
  msubu $ac2, $12, $13
  mfhi $14, $ac1
  mflo $15, $ac0
  mthi $16, $ac3
  mtlo $17, $ac2

  mult $2, $3
  multu $4, $5
  madd $6, $7
  maddu $8, $9
  msub $10, $11
  msubu $12, $13
  mfhi $14
  mflo $15
  mthi $16
  mtlo $17
