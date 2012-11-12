# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 | FileCheck %s
# Check that the assembler can handle the documented syntax
# for FPU instructions.
# CHECK: .section __TEXT,__text,regular,pure_instructions
#------------------------------------------------------------------------------
# FP aritmetic  instructions
#------------------------------------------------------------------------------

# CHECK:  abs.d      $f12, $f14         # encoding: [0x05,0x73,0x20,0x46]
# CHECK:  abs.s      $f6, $f7           # encoding: [0x85,0x39,0x00,0x46]
# CHECK:  add.d      $f8, $f12, $f14    # encoding: [0x00,0x62,0x2e,0x46]
# CHECK:  add.s      $f9, $f6, $f7      # encoding: [0x40,0x32,0x07,0x46]
# CHECK:  floor.w.d  $f12, $f14         # encoding: [0x0f,0x73,0x20,0x46]
# CHECK:  floor.w.s  $f6, $f7           # encoding: [0x8f,0x39,0x00,0x46]
# CHECK:  ceil.w.d   $f12, $f14         # encoding: [0x0e,0x73,0x20,0x46]
# CHECK:  ceil.w.s   $f6, $f7           # encoding: [0x8e,0x39,0x00,0x46]
# CHECK:  mul.d      $f8, $f12, $f14    # encoding: [0x02,0x62,0x2e,0x46]
# CHECK:  mul.s      $f9, $f6, $f7      # encoding: [0x42,0x32,0x07,0x46]
# CHECK:  neg.d      $f12, $f14         # encoding: [0x07,0x73,0x20,0x46]
# CHECK:  neg.s      $f6, $f7           # encoding: [0x87,0x39,0x00,0x46]
# CHECK:  round.w.d  $f12, $f14         # encoding: [0x0c,0x73,0x20,0x46]
# CHECK:  round.w.s  $f6, $f7           # encoding: [0x8c,0x39,0x00,0x46]
# CHECK:  sqrt.d     $f12, $f14         # encoding: [0x04,0x73,0x20,0x46]
# CHECK:  sqrt.s     $f6, $f7           # encoding: [0x84,0x39,0x00,0x46]
# CHECK:  sub.d      $f8, $f12, $f14    # encoding: [0x01,0x62,0x2e,0x46]
# CHECK:  sub.s      $f9, $f6, $f7      # encoding: [0x41,0x32,0x07,0x46]
# CHECK:  trunc.w.d  $f12, $f14         # encoding: [0x0d,0x73,0x20,0x46]
# CHECK:  trunc.w.s  $f6, $f7           # encoding: [0x8d,0x39,0x00,0x46]

    abs.d      $f12,$f14
    abs.s      $f6,$f7
    add.d      $f8,$f12,$f14
    add.s      $f9,$f6,$f7
    floor.w.d  $f12,$f14
    floor.w.s  $f6,$f7
    ceil.w.d   $f12,$f14
    ceil.w.s   $f6,$f7
    mul.d      $f8,$f12,$f14
    mul.s      $f9,$f6, $f7
    neg.d      $f12,$f14
    neg.s      $f6,$f7
    round.w.d  $f12,$f14
    round.w.s  $f6,$f7
    sqrt.d     $f12,$f14
    sqrt.s     $f6,$f7
    sub.d      $f8,$f12,$f14
    sub.s      $f9,$f6,$f7
    trunc.w.d  $f12,$f14
    trunc.w.s  $f6,$f7

#------------------------------------------------------------------------------
# FP compare instructions
#------------------------------------------------------------------------------

# CHECK:  c.eq.d    $f12, $f14        # encoding: [0x32,0x60,0x2e,0x46]
# CHECK:  c.eq.s    $f6, $f7          # encoding: [0x32,0x30,0x07,0x46]
# CHECK:  c.f.d     $f12, $f14        # encoding: [0x30,0x60,0x2e,0x46]
# CHECK:  c.f.s     $f6, $f7          # encoding: [0x30,0x30,0x07,0x46]
# CHECK:  c.le.d    $f12, $f14        # encoding: [0x3e,0x60,0x2e,0x46]
# CHECK:  c.le.s    $f6, $f7          # encoding: [0x3e,0x30,0x07,0x46]
# CHECK:  c.lt.d    $f12, $f14        # encoding: [0x3c,0x60,0x2e,0x46]
# CHECK:  c.lt.s    $f6, $f7          # encoding: [0x3c,0x30,0x07,0x46]
# CHECK:  c.nge.d   $f12, $f14        # encoding: [0x3d,0x60,0x2e,0x46]
# CHECK:  c.nge.s   $f6, $f7          # encoding: [0x3d,0x30,0x07,0x46]
# CHECK:  c.ngl.d   $f12, $f14        # encoding: [0x3b,0x60,0x2e,0x46]
# CHECK:  c.ngl.s   $f6, $f7          # encoding: [0x3b,0x30,0x07,0x46]
# CHECK:  c.ngle.d  $f12, $f14        # encoding: [0x39,0x60,0x2e,0x46]
# CHECK:  c.ngle.s  $f6, $f7          # encoding: [0x39,0x30,0x07,0x46]
# CHECK:  c.ngt.d   $f12, $f14        # encoding: [0x3f,0x60,0x2e,0x46]
# CHECK:  c.ngt.s   $f6, $f7          # encoding: [0x3f,0x30,0x07,0x46]
# CHECK:  c.ole.d   $f12, $f14        # encoding: [0x36,0x60,0x2e,0x46]
# CHECK:  c.ole.s   $f6, $f7          # encoding: [0x36,0x30,0x07,0x46]
# CHECK:  c.olt.d   $f12, $f14        # encoding: [0x34,0x60,0x2e,0x46]
# CHECK:  c.olt.s   $f6, $f7          # encoding: [0x34,0x30,0x07,0x46]
# CHECK:  c.seq.d   $f12, $f14        # encoding: [0x3a,0x60,0x2e,0x46]
# CHECK:  c.seq.s   $f6, $f7          # encoding: [0x3a,0x30,0x07,0x46]
# CHECK:  c.sf.d    $f12, $f14        # encoding: [0x38,0x60,0x2e,0x46]
# CHECK:  c.sf.s    $f6, $f7          # encoding: [0x38,0x30,0x07,0x46]
# CHECK:  c.ueq.d   $f12, $f14        # encoding: [0x33,0x60,0x2e,0x46]
# CHECK:  c.ueq.s   $f28, $f18        # encoding: [0x33,0xe0,0x12,0x46]
# CHECK:  c.ule.d   $f12, $f14        # encoding: [0x37,0x60,0x2e,0x46]
# CHECK:  c.ule.s   $f6, $f7          # encoding: [0x37,0x30,0x07,0x46]
# CHECK:  c.ult.d   $f12, $f14        # encoding: [0x35,0x60,0x2e,0x46]
# CHECK:  c.ult.s   $f6, $f7          # encoding: [0x35,0x30,0x07,0x46]
# CHECK:  c.un.d    $f12, $f14        # encoding: [0x31,0x60,0x2e,0x46]
# CHECK:  c.un.s    $f6, $f7          # encoding: [0x31,0x30,0x07,0x46]

     c.eq.d    $f12,$f14
     c.eq.s    $f6,$f7
     c.f.d     $f12,$f14
     c.f.s     $f6,$f7
     c.le.d    $f12,$f14
     c.le.s    $f6,$f7
     c.lt.d    $f12,$f14
     c.lt.s    $f6,$f7
     c.nge.d   $f12,$f14
     c.nge.s   $f6,$f7
     c.ngl.d   $f12,$f14
     c.ngl.s   $f6,$f7
     c.ngle.d  $f12,$f14
     c.ngle.s  $f6,$f7
     c.ngt.d   $f12,$f14
     c.ngt.s   $f6,$f7
     c.ole.d   $f12,$f14
     c.ole.s   $f6,$f7
     c.olt.d   $f12,$f14
     c.olt.s   $f6,$f7
     c.seq.d   $f12,$f14
     c.seq.s   $f6,$f7
     c.sf.d    $f12,$f14
     c.sf.s    $f6,$f7
     c.ueq.d   $f12,$f14
     c.ueq.s   $f28,$f18
     c.ule.d   $f12,$f14
     c.ule.s   $f6,$f7
     c.ult.d   $f12,$f14
     c.ult.s   $f6,$f7
     c.un.d    $f12,$f14
     c.un.s    $f6,$f7

#------------------------------------------------------------------------------
# FP convert instructions
#------------------------------------------------------------------------------
# CHECK:  cvt.d.s   $f6, $f7          # encoding: [0xa1,0x39,0x00,0x46]
# CHECK:  cvt.d.w   $f12, $f14        # encoding: [0x21,0x73,0x80,0x46]
# CHECK:  cvt.s.d   $f12, $f14        # encoding: [0x20,0x73,0x20,0x46]
# CHECK:  cvt.s.w   $f6, $f7          # encoding: [0xa0,0x39,0x80,0x46]
# CHECK:  cvt.w.d   $f12, $f14        # encoding: [0x24,0x73,0x20,0x46]
# CHECK:  cvt.w.s   $f6, $f7          # encoding: [0xa4,0x39,0x00,0x46]

  cvt.d.s   $f6,$f7
  cvt.d.w   $f12,$f14
  cvt.s.d   $f12,$f14
  cvt.s.w   $f6,$f7
  cvt.w.d   $f12,$f14
  cvt.w.s   $f6,$f7

#------------------------------------------------------------------------------
# FP move instructions
#------------------------------------------------------------------------------

# CHECK:  cfc1    $6, $fcc0            # encoding: [0x00,0x00,0x46,0x44]
# CHECK:  mfc1    $6, $f7              # encoding: [0x00,0x38,0x06,0x44]
# CHECK:  mfhi    $5                   # encoding: [0x10,0x28,0x00,0x00]
# CHECK:  mflo    $5                   # encoding: [0x12,0x28,0x00,0x00]
# CHECK:  mov.d   $f6, $f8             # encoding: [0x86,0x41,0x20,0x46]
# CHECK:  mov.s   $f6, $f7             # encoding: [0x86,0x39,0x00,0x46]
# CHECK:  mtc1    $6, $f7              # encoding: [0x00,0x38,0x86,0x44]
# CHECK:  mthi    $7                   # encoding: [0x11,0x00,0xe0,0x00]
# CHECK:  mtlo    $7                   # encoding: [0x13,0x00,0xe0,0x00]
# CHECK:  swc1    $f9, 9158($7)        # encoding: [0xc6,0x23,0xe9,0xe4]
# CHECK:  mfc0    $6, $7, 0               # encoding: [0x00,0x38,0x06,0x40]
# CHECK:  mtc0    $9, $8, 0               # encoding: [0x00,0x40,0x89,0x40]
# CHECK:  mfc2    $5, $7, 0               # encoding: [0x00,0x38,0x05,0x48]
# CHECK:  mtc2    $9, $4, 0               # encoding: [0x00,0x20,0x89,0x48]
# CHECK:  mfc0    $6, $7, 2               # encoding: [0x02,0x38,0x06,0x40]
# CHECK:  mtc0    $9, $8, 3               # encoding: [0x03,0x40,0x89,0x40]
# CHECK:  mfc2    $5, $7, 4               # encoding: [0x04,0x38,0x05,0x48]
# CHECK:  mtc2    $9, $4, 5               # encoding: [0x05,0x20,0x89,0x48]

   cfc1    $a2,$0
   mfc1    $a2,$f7
   mfhi    $a1
   mflo    $a1
   mov.d   $f6,$f8
   mov.s   $f6,$f7
   mtc1    $a2,$f7
   mthi    $a3
   mtlo    $a3
   swc1    $f9,9158($a3)
   mfc0    $6, $7
   mtc0    $9, $8
   mfc2    $5, $7
   mtc2    $9, $4
   mfc0    $6, $7, 2
   mtc0    $9, $8, 3
   mfc2    $5, $7, 4
   mtc2    $9, $4, 5
