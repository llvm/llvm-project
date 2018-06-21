# RUN: llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips \
# RUN: -mcpu=mips32r2 | FileCheck -check-prefix=CHECK-EL %s
# RUN: llvm-mc %s -triple=mips -show-encoding -mattr=micromips \
# RUN: -mcpu=mips32r2 | FileCheck -check-prefix=CHECK-EB %s
# Check that the assembler can handle the documented syntax
# for fpu instructions
#------------------------------------------------------------------------------
# FPU Instructions
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL: add.s      $f4, $f6, $f8    # encoding: [0x06,0x55,0x30,0x20]
# CHECK-EL: add.d      $f4, $f6, $f8    # encoding: [0x06,0x55,0x30,0x21]
# CHECK-EL: div.s      $f4, $f6, $f8    # encoding: [0x06,0x55,0xf0,0x20]
# CHECK-EL: div.d      $f4, $f6, $f8    # encoding: [0x06,0x55,0xf0,0x21]
# CHECK-EL: mul.s      $f4, $f6, $f8    # encoding: [0x06,0x55,0xb0,0x20]
# CHECK-EL: mul.d      $f4, $f6, $f8    # encoding: [0x06,0x55,0xb0,0x21]
# CHECK-EL: sub.s      $f4, $f6, $f8    # encoding: [0x06,0x55,0x70,0x20]
# CHECK-EL: sub.d      $f4, $f6, $f8    # encoding: [0x06,0x55,0x70,0x21]
# CHECK-EL: lwc1       $f2, 4($6)       # encoding: [0x46,0x9c,0x04,0x00]
# CHECK-EL: ldc1       $f2, 4($6)       # encoding: [0x46,0xbc,0x04,0x00]
# CHECK-EL: swc1       $f2, 4($6)       # encoding: [0x46,0x98,0x04,0x00]
# CHECK-EL: sdc1       $f2, 4($6)       # encoding: [0x46,0xb8,0x04,0x00]
# CHECK-EL: bc1f       1332             # encoding: [0x80,0x43,0x9a,0x02]
# CHECK-EL: nop                         # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EL: bc1t       1332             # encoding: [0xa0,0x43,0x9a,0x02]
# CHECK-EL: nop                         # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EL: luxc1      $f2, $4($6)      # encoding: [0x86,0x54,0x48,0x11]
# CHECK-EL: suxc1      $f2, $4($6)      # encoding: [0x86,0x54,0x88,0x11]
# CHECK-EL: ceil.w.s   $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x1b]
# CHECK-EL: ceil.w.d   $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x5b]
# CHECK-EL: cvt.w.s    $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x09]
# CHECK-EL: cvt.w.d    $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x49]
# CHECK-EL: floor.w.s  $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x0b]
# CHECK-EL: floor.w.d  $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x4b]
# CHECK-EL: round.w.s  $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x3b]
# CHECK-EL: round.w.d  $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x7b]
# CHECK-EL: sqrt.s     $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x0a]
# CHECK-EL: sqrt.d     $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x4a]
# CHECK-EL: trunc.w.s  $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x2b]
# CHECK-EL: trunc.w.d  $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x6b]
# CHECK-EL: abs.s      $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x03]
# CHECK-EL: abs.d      $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x23]
# CHECK-EL: mov.s      $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x00]
# CHECK-EL: mov.d      $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x20]
# CHECK-EL: neg.s      $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x0b]
# CHECK-EL: neg.d      $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x2b]
# CHECK-EL: cvt.d.s    $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x13]
# CHECK-EL: cvt.d.w    $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x33]
# CHECK-EL: cvt.s.d    $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x1b]
# CHECK-EL: cvt.s.w    $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x3b]
# CHECK-EL: cfc1    $6, $0              # encoding: [0xc0,0x54,0x3b,0x10]
# CHECK-EL: ctc1    $6, $0              # encoding: [0xc0,0x54,0x3b,0x18]
# CHECK-EL: mfc1    $6, $f8             # encoding: [0xc8,0x54,0x3b,0x20]
# CHECK-EL: mtc1    $6, $f8             # encoding: [0xc8,0x54,0x3b,0x28]
# CHECK-EL: mfhc1   $6, $f8             # encoding: [0xc8,0x54,0x3b,0x30]
# CHECK-EL: mthc1   $6, $f8             # encoding: [0xc8,0x54,0x3b,0x38]
# CHECK-EL: movz.s  $f4, $f6, $7        # encoding: [0xe6,0x54,0x78,0x20]
# CHECK-EL: movz.d  $f4, $f6, $7        # encoding: [0xe6,0x54,0x78,0x21]
# CHECK-EL: movn.s  $f4, $f6, $7        # encoding: [0xe6,0x54,0x38,0x20]
# CHECK-EL: movn.d  $f4, $f6, $7        # encoding: [0xe6,0x54,0x38,0x21]
# CHECK-EL: movt.s  $f4, $f6, $fcc0     # encoding: [0x86,0x54,0x60,0x00]
# CHECK-EL: movt.d  $f4, $f6, $fcc0     # encoding: [0x86,0x54,0x60,0x02]
# CHECK-EL: movf.s  $f4, $f6, $fcc0     # encoding: [0x86,0x54,0x20,0x00]
# CHECK-EL: movf.d  $f4, $f6, $fcc0     # encoding: [0x86,0x54,0x20,0x02]
# CHECK-EL: madd.s  $f2, $f4, $f6, $f8  # encoding: [0x06,0x55,0x01,0x11]
# CHECK-EL: madd.d  $f2, $f4, $f6, $f8  # encoding: [0x06,0x55,0x09,0x11]
# CHECK-EL: msub.s  $f2, $f4, $f6, $f8  # encoding: [0x06,0x55,0x21,0x11]
# CHECK-EL: msub.d  $f2, $f4, $f6, $f8  # encoding: [0x06,0x55,0x29,0x11]
# CHECK-EL: nmadd.s $f2, $f4, $f6, $f8  # encoding: [0x06,0x55,0x02,0x11]
# CHECK-EL: nmadd.d $f2, $f4, $f6, $f8  # encoding: [0x06,0x55,0x0a,0x11]
# CHECK-EL: nmsub.s $f2, $f4, $f6, $f8  # encoding: [0x06,0x55,0x22,0x11]
# CHECK-EL: nmsub.d $f2, $f4, $f6, $f8  # encoding: [0x06,0x55,0x2a,0x11]
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB: add.s $f4, $f6, $f8         # encoding: [0x55,0x06,0x20,0x30]
# CHECK-EB: add.d $f4, $f6, $f8         # encoding: [0x55,0x06,0x21,0x30]
# CHECK-EB: div.s $f4, $f6, $f8         # encoding: [0x55,0x06,0x20,0xf0]
# CHECK-EB: div.d $f4, $f6, $f8         # encoding: [0x55,0x06,0x21,0xf0]
# CHECK-EB: mul.s $f4, $f6, $f8         # encoding: [0x55,0x06,0x20,0xb0]
# CHECK-EB: mul.d $f4, $f6, $f8         # encoding: [0x55,0x06,0x21,0xb0]
# CHECK-EB: sub.s $f4, $f6, $f8         # encoding: [0x55,0x06,0x20,0x70]
# CHECK-EB: sub.d $f4, $f6, $f8         # encoding: [0x55,0x06,0x21,0x70]
# CHECK-EB: lwc1  $f2, 4($6)            # encoding: [0x9c,0x46,0x00,0x04]
# CHECK-EB: ldc1  $f2, 4($6)            # encoding: [0xbc,0x46,0x00,0x04]
# CHECK-EB: swc1  $f2, 4($6)            # encoding: [0x98,0x46,0x00,0x04]
# CHECK-EB: sdc1  $f2, 4($6)            # encoding: [0xb8,0x46,0x00,0x04]
# CHECK-EB: bc1f  1332                  # encoding: [0x43,0x80,0x02,0x9a]
# CHECK-EB: nop                         # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EB: bc1t  1332                  # encoding: [0x43,0xa0,0x02,0x9a]
# CHECK-EB: nop                         # encoding: [0x00,0x00,0x00,0x00]
# CHECK-EB: luxc1 $f2, $4($6)           # encoding: [0x54,0x86,0x11,0x48]
# CHECK-EB: suxc1 $f2, $4($6)           # encoding: [0x54,0x86,0x11,0x88]
# CHECK-EB: ceil.w.s  $f6, $f8          # encoding: [0x54,0xc8,0x1b,0x3b]
# CHECK-EB: ceil.w.d  $f6, $f8          # encoding: [0x54,0xc8,0x5b,0x3b]
# CHECK-EB: cvt.w.s   $f6, $f8          # encoding: [0x54,0xc8,0x09,0x3b]
# CHECK-EB: cvt.w.d   $f6, $f8          # encoding: [0x54,0xc8,0x49,0x3b]
# CHECK-EB: floor.w.s $f6, $f8          # encoding: [0x54,0xc8,0x0b,0x3b]
# CHECK-EB: floor.w.d $f6, $f8          # encoding: [0x54,0xc8,0x4b,0x3b]
# CHECK-EB: round.w.s $f6, $f8          # encoding: [0x54,0xc8,0x3b,0x3b]
# CHECK-EB: round.w.d $f6, $f8          # encoding: [0x54,0xc8,0x7b,0x3b]
# CHECK-EB: sqrt.s    $f6, $f8          # encoding: [0x54,0xc8,0x0a,0x3b]
# CHECK-EB: sqrt.d    $f6, $f8          # encoding: [0x54,0xc8,0x4a,0x3b]
# CHECK-EB: trunc.w.s $f6, $f8          # encoding: [0x54,0xc8,0x2b,0x3b]
# CHECK-EB: trunc.w.d $f6, $f8          # encoding: [0x54,0xc8,0x6b,0x3b]
# CHECK-EB: abs.s $f6, $f8              # encoding: [0x54,0xc8,0x03,0x7b]
# CHECK-EB: abs.d $f6, $f8              # encoding: [0x54,0xc8,0x23,0x7b]
# CHECK-EB: mov.s $f6, $f8              # encoding: [0x54,0xc8,0x00,0x7b]
# CHECK-EB: mov.d $f6, $f8              # encoding: [0x54,0xc8,0x20,0x7b]
# CHECK-EB: neg.s $f6, $f8              # encoding: [0x54,0xc8,0x0b,0x7b]
# CHECK-EB: neg.d $f6, $f8              # encoding: [0x54,0xc8,0x2b,0x7b]
# CHECK-EB: cvt.d.s $f6, $f8            # encoding: [0x54,0xc8,0x13,0x7b]
# CHECK-EB: cvt.d.w $f6, $f8            # encoding: [0x54,0xc8,0x33,0x7b]
# CHECK-EB: cvt.s.d $f6, $f8            # encoding: [0x54,0xc8,0x1b,0x7b]
# CHECK-EB: cvt.s.w $f6, $f8            # encoding: [0x54,0xc8,0x3b,0x7b]
# CHECK-EB: cfc1    $6, $0              # encoding: [0x54,0xc0,0x10,0x3b]
# CHECK-EB: ctc1    $6, $0              # encoding: [0x54,0xc0,0x18,0x3b]
# CHECK-EB: mfc1    $6, $f8             # encoding: [0x54,0xc8,0x20,0x3b]
# CHECK-EB: mtc1    $6, $f8             # encoding: [0x54,0xc8,0x28,0x3b]
# CHECK-EB: mfhc1   $6, $f8             # encoding: [0x54,0xc8,0x30,0x3b]
# CHECK-EB: mthc1   $6, $f8             # encoding: [0x54,0xc8,0x38,0x3b]
# CHECK-EB: movz.s  $f4, $f6, $7        # encoding: [0x54,0xe6,0x20,0x78]
# CHECK-EB: movz.d  $f4, $f6, $7        # encoding: [0x54,0xe6,0x21,0x78]
# CHECK-EB: movn.s  $f4, $f6, $7        # encoding: [0x54,0xe6,0x20,0x38]
# CHECK-EB: movn.d  $f4, $f6, $7        # encoding: [0x54,0xe6,0x21,0x38]
# CHECK-EB: movt.s  $f4, $f6, $fcc0     # encoding: [0x54,0x86,0x00,0x60]
# CHECK-EB: movt.d  $f4, $f6, $fcc0     # encoding: [0x54,0x86,0x02,0x60]
# CHECK-EB: movf.s  $f4, $f6, $fcc0     # encoding: [0x54,0x86,0x00,0x20]
# CHECK-EB: movf.d  $f4, $f6, $fcc0     # encoding: [0x54,0x86,0x02,0x20]
# CHECK-EB: madd.s  $f2, $f4, $f6, $f8  # encoding: [0x55,0x06,0x11,0x01]
# CHECK-EB: madd.d  $f2, $f4, $f6, $f8  # encoding: [0x55,0x06,0x11,0x09]
# CHECK-EB: msub.s  $f2, $f4, $f6, $f8  # encoding: [0x55,0x06,0x11,0x21]
# CHECK-EB: msub.d  $f2, $f4, $f6, $f8  # encoding: [0x55,0x06,0x11,0x29]
# CHECK-EB: nmadd.s $f2, $f4, $f6, $f8  # encoding: [0x55,0x06,0x11,0x02]
# CHECK-EB: nmadd.d $f2, $f4, $f6, $f8  # encoding: [0x55,0x06,0x11,0x0a]
# CHECK-EB: nmsub.s $f2, $f4, $f6, $f8  # encoding: [0x55,0x06,0x11,0x22]
# CHECK-EB: nmsub.d $f2, $f4, $f6, $f8  # encoding: [0x55,0x06,0x11,0x2a]

    add.s      $f4, $f6, $f8
    add.d      $f4, $f6, $f8
    div.s      $f4, $f6, $f8
    div.d      $f4, $f6, $f8
    mul.s      $f4, $f6, $f8
    mul.d      $f4, $f6, $f8
    sub.s      $f4, $f6, $f8
    sub.d      $f4, $f6, $f8
    lwc1       $f2, 4($6)
    ldc1       $f2, 4($6)
    swc1       $f2, 4($6)
    sdc1       $f2, 4($6)
    bc1f       1332
    bc1t       1332
    luxc1      $f2, $4($6)
    suxc1      $f2, $4($6)
    ceil.w.s   $f6, $f8
    ceil.w.d   $f6, $f8
    cvt.w.s    $f6, $f8
    cvt.w.d    $f6, $f8
    floor.w.s  $f6, $f8
    floor.w.d  $f6, $f8
    round.w.s  $f6, $f8
    round.w.d  $f6, $f8
    sqrt.s     $f6, $f8
    sqrt.d     $f6, $f8
    trunc.w.s  $f6, $f8
    trunc.w.d  $f6, $f8
    abs.s      $f6, $f8
    abs.d      $f6, $f8
    mov.s      $f6, $f8
    mov.d      $f6, $f8
    neg.s      $f6, $f8
    neg.d      $f6, $f8
    cvt.d.s    $f6, $f8
    cvt.d.w    $f6, $f8
    cvt.s.d    $f6, $f8
    cvt.s.w    $f6, $f8
    cfc1       $6, $0
    ctc1       $6, $0
    mfc1       $6, $f8
    mtc1       $6, $f8
    mfhc1      $6, $f8
    mthc1      $6, $f8
    movz.s     $f4, $f6, $7
    movz.d     $f4, $f6, $7
    movn.s     $f4, $f6, $7
    movn.d     $f4, $f6, $7
    movt.s     $f4, $f6, $fcc0
    movt.d     $f4, $f6, $fcc0
    movf.s     $f4, $f6, $fcc0
    movf.d     $f4, $f6, $fcc0
    madd.s     $f2, $f4, $f6, $f8
    madd.d     $f2, $f4, $f6, $f8
    msub.s     $f2, $f4, $f6, $f8
    msub.d     $f2, $f4, $f6, $f8
    nmadd.s    $f2, $f4, $f6, $f8
    nmadd.d    $f2, $f4, $f6, $f8
    nmsub.s    $f2, $f4, $f6, $f8
    nmsub.d    $f2, $f4, $f6, $f8
