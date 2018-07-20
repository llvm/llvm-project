# RUN: llvm-mc %s -triple=mipsel -show-encoding -show-inst -mattr=micromips \
# RUN: -mcpu=mips32r2 | FileCheck -check-prefix=CHECK-EL %s
# RUN: llvm-mc %s -triple=mips -show-encoding -show-inst -mattr=micromips \
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
# CHECK-EL: ceil.w.s   $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x1b]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} CEIL_W_S_MM
# CHECK-EL: ceil.w.d   $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x5b]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} CEIL_W_MM
# CHECK-EL: cvt.w.s    $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x09]
# CHECK-EL: cvt.w.d    $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x49]
# CHECK-EL: floor.w.s  $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x0b]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} FLOOR_W_S_MM
# CHECK-EL: floor.w.d  $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x4b]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} FLOOR_W_MM
# CHECK-EL: round.w.s  $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x3b]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} ROUND_W_S_MM
# CHECK-EL: round.w.d  $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x7b]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} ROUND_W_MM
# CHECK-EL: sqrt.s     $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x0a]
# CHECK-EL: sqrt.d     $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x4a]
# CHECK-EL: trunc.w.s  $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x2b]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} TRUNC_W_S_MM
# CHECK-EL: trunc.w.d  $f6, $f8         # encoding: [0xc8,0x54,0x3b,0x6b]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} TRUNC_W_MM
# CHECK-EL: abs.s      $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x03]
# CHECK-EL: abs.d      $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x23]
# CHECK-EL: mov.s      $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x00]
# CHECK-EL: mov.d      $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x20]
# CHECK-EL: neg.s      $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x0b]
# CHECK-EL: neg.d      $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x2b]
# CHECK-EL: cvt.d.s    $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x13]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} CVT_D32_S_MM
# CHECK-EL: cvt.d.w    $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x33]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} CVT_D32_W_MM
# CHECK-EL: cvt.s.d    $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x1b]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} CVT_S_D32_MM
# CHECK-EL: cvt.s.w    $f6, $f8         # encoding: [0xc8,0x54,0x7b,0x3b]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} CVT_S_W_MM
# CHECK-EL: cfc1    $6, $0              # encoding: [0xc0,0x54,0x3b,0x10]
# CHECK-EL: ctc1    $6, $0              # encoding: [0xc0,0x54,0x3b,0x18]
# CHECK-EL: mfc1    $6, $f8             # encoding: [0xc8,0x54,0x3b,0x20]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} MFC1_MM
# CHECK-EL: mtc1    $6, $f8             # encoding: [0xc8,0x54,0x3b,0x28]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} MTC1_MM
# CHECK-EL: mfhc1   $6, $f8             # encoding: [0xc8,0x54,0x3b,0x30]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} MFHC1_D32_MM
# CHECK-EL: mthc1   $6, $f8             # encoding: [0xc8,0x54,0x3b,0x38]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} MTHC1_D32_MM
# CHECK-EL: movz.s  $f4, $f6, $7        # encoding: [0xe6,0x54,0x78,0x20]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} MOVZ_I_S_MM
# CHECK-EL: movz.d  $f4, $f6, $7        # encoding: [0xe6,0x54,0x78,0x21]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} MOVZ_I_D32_MM
# CHECK-EL: movn.s  $f4, $f6, $7        # encoding: [0xe6,0x54,0x38,0x20]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} MOVN_I_S_MM
# CHECK-EL: movn.d  $f4, $f6, $7        # encoding: [0xe6,0x54,0x38,0x21]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} MOVN_I_D32_MM
# CHECK-EL: movt.s  $f4, $f6, $fcc0     # encoding: [0x86,0x54,0x60,0x00]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} MOVT_S_MM
# CHECK-EL: movt.d  $f4, $f6, $fcc0     # encoding: [0x86,0x54,0x60,0x02]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} MOVT_D32_MM
# CHECK-EL: movf.s  $f4, $f6, $fcc0     # encoding: [0x86,0x54,0x20,0x00]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} MOVF_S_MM
# CHECK-EL: movf.d  $f4, $f6, $fcc0     # encoding: [0x86,0x54,0x20,0x02]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} MOVF_D32_MM
# CHECK-EL: madd.s  $f2, $f4, $f6, $f8  # encoding: [0x06,0x55,0x01,0x11]
# CHECK-EL: madd.d  $f2, $f4, $f6, $f8  # encoding: [0x06,0x55,0x09,0x11]
# CHECK-EL: msub.s  $f2, $f4, $f6, $f8  # encoding: [0x06,0x55,0x21,0x11]
# CHECK-EL: msub.d  $f2, $f4, $f6, $f8  # encoding: [0x06,0x55,0x29,0x11]
# CHECK-EL: nmadd.s $f2, $f4, $f6, $f8  # encoding: [0x06,0x55,0x02,0x11]
# CHECK-EL: nmadd.d $f2, $f4, $f6, $f8  # encoding: [0x06,0x55,0x0a,0x11]
# CHECK-EL: nmsub.s $f2, $f4, $f6, $f8  # encoding: [0x06,0x55,0x22,0x11]
# CHECK-EL: nmsub.d $f2, $f4, $f6, $f8  # encoding: [0x06,0x55,0x2a,0x11]
# CHECK-EL: c.f.s $f6, $f7              # encoding: [0xe6,0x54,0x3c,0x00]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_F_S_MM
# CHECK-EL: c.un.s  $f6, $f7            # encoding: [0xe6,0x54,0x7c,0x00]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_UN_S_MM
# CHECK-EL: c.eq.s  $f6, $f7            # encoding: [0xe6,0x54,0xbc,0x00]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_EQ_S_MM
# CHECK-EL: c.olt.s $f6, $f7            # encoding: [0xe6,0x54,0x3c,0x01]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_OLT_S_MM
# CHECK-EL: c.ult.s $f6, $f7            # encoding: [0xe6,0x54,0x7c,0x01]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_ULT_S_MM
# CHECK-EL: c.ole.s $f6, $f7            # encoding: [0xe6,0x54,0xbc,0x01]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_OLE_S_MM
# CHECK-EL: c.ule.s $f6, $f7            # encoding: [0xe6,0x54,0xfc,0x01]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_ULE_S_MM
# CHECK-EL: c.sf.s  $f6, $f7            # encoding: [0xe6,0x54,0x3c,0x02]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_SF_S_MM
# CHECK-EL: c.ngle.s  $f6, $f7          # encoding: [0xe6,0x54,0x7c,0x02]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_NGLE_S_MM
# CHECK-EL: c.seq.s $f6, $f7            # encoding: [0xe6,0x54,0xbc,0x02]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_SEQ_S_MM
# CHECK-EL: c.ngl.s $f6, $f7            # encoding: [0xe6,0x54,0xfc,0x02]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_NGL_S_MM
# CHECK-EL: c.lt.s  $f6, $f7            # encoding: [0xe6,0x54,0x3c,0x03]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_LT_S_MM
# CHECK-EL: c.nge.s $f6, $f7            # encoding: [0xe6,0x54,0x7c,0x03]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_NGE_S_MM
# CHECK-EL: c.le.s  $f6, $f7            # encoding: [0xe6,0x54,0xbc,0x03]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_LE_S_MM
# CHECK-EL: c.ngt.s $f6, $f7            # encoding: [0xe6,0x54,0xfc,0x03]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_NGT_S_MM
# CHECK-EL: c.sf.d  $f30, $f0           # encoding: [0x1e,0x54,0x3c,0x06]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_SF_D32_MM
# CHECK-EL: c.f.d $f12, $f14            # encoding: [0xcc,0x55,0x3c,0x04]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_F_D32_MM
# CHECK-EL: c.un.d  $f12, $f14          # encoding: [0xcc,0x55,0x7c,0x04]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_UN_D32_MM
# CHECK-EL: c.eq.d  $f12, $f14          # encoding: [0xcc,0x55,0xbc,0x04]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_EQ_D32_MM
# CHECK-EL: c.ueq.d $f12, $f14          # encoding: [0xcc,0x55,0xfc,0x04]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_UEQ_D32_MM
# CHECK-EL: c.olt.d $f12, $f14          # encoding: [0xcc,0x55,0x3c,0x05]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_OLT_D32_MM
# CHECK-EL: c.ult.d $f12, $f14          # encoding: [0xcc,0x55,0x7c,0x05]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_ULT_D32_MM
# CHECK-EL: c.ole.d $f12, $f14          # encoding: [0xcc,0x55,0xbc,0x05]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_OLE_D32_MM
# CHECK-EL: c.ule.d $f12, $f14          # encoding: [0xcc,0x55,0xfc,0x05]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_ULE_D32_MM
# CHECK-EL: c.sf.d  $f12, $f14          # encoding: [0xcc,0x55,0x3c,0x06]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_SF_D32_MM
# CHECK-EL: c.ngle.d  $f12, $f14        # encoding: [0xcc,0x55,0x7c,0x06]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_NGLE_D32_MM
# CHECK-EL: c.seq.d $f12, $f14          # encoding: [0xcc,0x55,0xbc,0x06]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_SEQ_D32_MM
# CHECK-EL: c.ngl.d $f12, $f14          # encoding: [0xcc,0x55,0xfc,0x06]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_NGL_D32_MM
# CHECK-EL: c.lt.d  $f12, $f14          # encoding: [0xcc,0x55,0x3c,0x07]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_LT_D32_MM
# CHECK-EL: c.nge.d $f12, $f14          # encoding: [0xcc,0x55,0x7c,0x07]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_NGE_D32_MM
# CHECK-EL: c.le.d  $f12, $f14          # encoding: [0xcc,0x55,0xbc,0x07]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_LE_D32_MM
# CHECK-EL: c.ngt.d $f12, $f14          # encoding: [0xcc,0x55,0xfc,0x07]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} C_NGT_D32_MM
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
# CHECK-EB: ceil.w.s  $f6, $f8          # encoding: [0x54,0xc8,0x1b,0x3b]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} CEIL_W_S_MM
# CHECK-EB: ceil.w.d  $f6, $f8          # encoding: [0x54,0xc8,0x5b,0x3b]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} CEIL_W_MM
# CHECK-EB: cvt.w.s   $f6, $f8          # encoding: [0x54,0xc8,0x09,0x3b]
# CHECK-EB: cvt.w.d   $f6, $f8          # encoding: [0x54,0xc8,0x49,0x3b]
# CHECK-EB: floor.w.s $f6, $f8          # encoding: [0x54,0xc8,0x0b,0x3b]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} FLOOR_W_S_MM
# CHECK-EB: floor.w.d $f6, $f8          # encoding: [0x54,0xc8,0x4b,0x3b]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} FLOOR_W_MM
# CHECK-EB: round.w.s $f6, $f8          # encoding: [0x54,0xc8,0x3b,0x3b]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} ROUND_W_S_MM
# CHECK-EB: round.w.d $f6, $f8          # encoding: [0x54,0xc8,0x7b,0x3b]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} ROUND_W_MM
# CHECK-EB: sqrt.s    $f6, $f8          # encoding: [0x54,0xc8,0x0a,0x3b]
# CHECK-EB: sqrt.d    $f6, $f8          # encoding: [0x54,0xc8,0x4a,0x3b]
# CHECK-EB: trunc.w.s $f6, $f8          # encoding: [0x54,0xc8,0x2b,0x3b]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} TRUNC_W_S_MM
# CHECK-EB: trunc.w.d $f6, $f8          # encoding: [0x54,0xc8,0x6b,0x3b]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} TRUNC_W_MM
# CHECK-EB: abs.s $f6, $f8              # encoding: [0x54,0xc8,0x03,0x7b]
# CHECK-EB: abs.d $f6, $f8              # encoding: [0x54,0xc8,0x23,0x7b]
# CHECK-EB: mov.s $f6, $f8              # encoding: [0x54,0xc8,0x00,0x7b]
# CHECK-EB: mov.d $f6, $f8              # encoding: [0x54,0xc8,0x20,0x7b]
# CHECK-EB: neg.s $f6, $f8              # encoding: [0x54,0xc8,0x0b,0x7b]
# CHECK-EB: neg.d $f6, $f8              # encoding: [0x54,0xc8,0x2b,0x7b]
# CHECK-EB: cvt.d.s $f6, $f8            # encoding: [0x54,0xc8,0x13,0x7b]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} CVT_D32_S_MM
# CHECK-EB: cvt.d.w $f6, $f8            # encoding: [0x54,0xc8,0x33,0x7b]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} CVT_D32_W_MM
# CHECK-EB: cvt.s.d $f6, $f8            # encoding: [0x54,0xc8,0x1b,0x7b]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} CVT_S_D32_MM
# CHECK-EB: cvt.s.w $f6, $f8            # encoding: [0x54,0xc8,0x3b,0x7b]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} CVT_S_W_MM
# CHECK-EB: cfc1    $6, $0              # encoding: [0x54,0xc0,0x10,0x3b]
# CHECK-EB: ctc1    $6, $0              # encoding: [0x54,0xc0,0x18,0x3b]
# CHECK-EB: mfc1    $6, $f8             # encoding: [0x54,0xc8,0x20,0x3b]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} MFC1_MM
# CHECK-EB: mtc1    $6, $f8             # encoding: [0x54,0xc8,0x28,0x3b]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} MTC1_MM
# CHECK-EB: mfhc1   $6, $f8             # encoding: [0x54,0xc8,0x30,0x3b]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} MFHC1_D32_MM
# CHECK-EB: mthc1   $6, $f8             # encoding: [0x54,0xc8,0x38,0x3b]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} MTHC1_D32_MM
# CHECK-EB: movz.s  $f4, $f6, $7        # encoding: [0x54,0xe6,0x20,0x78]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} MOVZ_I_S_MM
# CHECK-EB: movz.d  $f4, $f6, $7        # encoding: [0x54,0xe6,0x21,0x78]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} MOVZ_I_D32_MM
# CHECK-EB: movn.s  $f4, $f6, $7        # encoding: [0x54,0xe6,0x20,0x38]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} MOVN_I_S_MM
# CHECK-EB: movn.d  $f4, $f6, $7        # encoding: [0x54,0xe6,0x21,0x38]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} MOVN_I_D32_MM
# CHECK-EB: movt.s  $f4, $f6, $fcc0     # encoding: [0x54,0x86,0x00,0x60]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} MOVT_S_MM
# CHECK-EB: movt.d  $f4, $f6, $fcc0     # encoding: [0x54,0x86,0x02,0x60]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} MOVT_D32_MM
# CHECK-EB: movf.s  $f4, $f6, $fcc0     # encoding: [0x54,0x86,0x00,0x20]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} MOVF_S_MM
# CHECK-EB: movf.d  $f4, $f6, $fcc0     # encoding: [0x54,0x86,0x02,0x20]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} MOVF_D32_MM
# CHECK-EB: madd.s  $f2, $f4, $f6, $f8  # encoding: [0x55,0x06,0x11,0x01]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} MADD_S_MM
# CHECK-EB: madd.d  $f2, $f4, $f6, $f8  # encoding: [0x55,0x06,0x11,0x09]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} MADD_D32_MM
# CHECK-EB: msub.s  $f2, $f4, $f6, $f8  # encoding: [0x55,0x06,0x11,0x21]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} MSUB_S_MM
# CHECK-EB: msub.d  $f2, $f4, $f6, $f8  # encoding: [0x55,0x06,0x11,0x29]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} MSUB_D32_MM
# CHECK-EB: nmadd.s $f2, $f4, $f6, $f8  # encoding: [0x55,0x06,0x11,0x02]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} NMADD_S_MM
# CHECK-EB: nmadd.d $f2, $f4, $f6, $f8  # encoding: [0x55,0x06,0x11,0x0a]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} NMADD_D32_MM
# CHECK-EB: nmsub.s $f2, $f4, $f6, $f8  # encoding: [0x55,0x06,0x11,0x22]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} NMSUB_S_MM
# CHECK-EB: nmsub.d $f2, $f4, $f6, $f8  # encoding: [0x55,0x06,0x11,0x2a]
# CHECK-EB: c.f.s $f6, $f7              # encoding: [0x54,0xe6,0x00,0x3c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_F_S_MM
# CHECK-EB: c.un.s  $f6, $f7            # encoding: [0x54,0xe6,0x00,0x7c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_UN_S_MM
# CHECK-EB: c.eq.s  $f6, $f7            # encoding: [0x54,0xe6,0x00,0xbc]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_EQ_S_MM
# CHECK-EB: c.olt.s $f6, $f7            # encoding: [0x54,0xe6,0x01,0x3c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_OLT_S_MM
# CHECK-EB: c.ult.s $f6, $f7            # encoding: [0x54,0xe6,0x01,0x7c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_ULT_S_MM
# CHECK-EB: c.ole.s $f6, $f7            # encoding: [0x54,0xe6,0x01,0xbc]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_OLE_S_MM
# CHECK-EB: c.ule.s $f6, $f7            # encoding: [0x54,0xe6,0x01,0xfc]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_ULE_S_MM
# CHECK-EB: c.sf.s  $f6, $f7            # encoding: [0x54,0xe6,0x02,0x3c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_SF_S_MM
# CHECK-EB: c.ngle.s  $f6, $f7          # encoding: [0x54,0xe6,0x02,0x7c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_NGLE_S_MM
# CHECK-EB: c.seq.s $f6, $f7            # encoding: [0x54,0xe6,0x02,0xbc]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_SEQ_S_MM
# CHECK-EB: c.ngl.s $f6, $f7            # encoding: [0x54,0xe6,0x02,0xfc]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_NGL_S_MM
# CHECK-EB: c.lt.s  $f6, $f7            # encoding: [0x54,0xe6,0x03,0x3c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_LT_S_MM
# CHECK-EB: c.nge.s $f6, $f7            # encoding: [0x54,0xe6,0x03,0x7c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_NGE_S_MM
# CHECK-EB: c.le.s  $f6, $f7            # encoding: [0x54,0xe6,0x03,0xbc]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_LE_S_MM
# CHECK-EB: c.ngt.s $f6, $f7            # encoding: [0x54,0xe6,0x03,0xfc]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_NGT_S_MM
# CHECK-EB: c.sf.d  $f30, $f0           # encoding: [0x54,0x1e,0x06,0x3c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_SF_D32_MM
# CHECK-EB: c.f.d $f12, $f14            # encoding: [0x55,0xcc,0x04,0x3c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_F_D32_MM
# CHECK-EB: c.un.d  $f12, $f14          # encoding: [0x55,0xcc,0x04,0x7c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_UN_D32_MM
# CHECK-EB: c.eq.d  $f12, $f14          # encoding: [0x55,0xcc,0x04,0xbc]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_EQ_D32_MM
# CHECK-EB: c.ueq.d $f12, $f14          # encoding: [0x55,0xcc,0x04,0xfc]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_UEQ_D32_MM
# CHECK-EB: c.olt.d $f12, $f14          # encoding: [0x55,0xcc,0x05,0x3c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_OLT_D32_MM
# CHECK-EB: c.ult.d $f12, $f14          # encoding: [0x55,0xcc,0x05,0x7c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_ULT_D32_MM
# CHECK-EB: c.ole.d $f12, $f14          # encoding: [0x55,0xcc,0x05,0xbc]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_OLE_D32_MM
# CHECK-EB: c.ule.d $f12, $f14          # encoding: [0x55,0xcc,0x05,0xfc]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_ULE_D32_MM
# CHECK-EB: c.sf.d  $f12, $f14          # encoding: [0x55,0xcc,0x06,0x3c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_SF_D32_MM
# CHECK-EB: c.ngle.d  $f12, $f14        # encoding: [0x55,0xcc,0x06,0x7c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_NGLE_D32_MM
# CHECK-EB: c.seq.d $f12, $f14          # encoding: [0x55,0xcc,0x06,0xbc]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_SEQ_D32_MM
# CHECK-EB: c.ngl.d $f12, $f14          # encoding: [0x55,0xcc,0x06,0xfc]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_NGL_D32_MM
# CHECK-EB: c.lt.d  $f12, $f14          # encoding: [0x55,0xcc,0x07,0x3c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_LT_D32_MM
# CHECK-EB: c.nge.d $f12, $f14          # encoding: [0x55,0xcc,0x07,0x7c]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_NGE_D32_MM
# CHECK-EB: c.le.d  $f12, $f14          # encoding: [0x55,0xcc,0x07,0xbc]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_LE_D32_MM
# CHECK-EB: c.ngt.d $f12, $f14          # encoding: [0x55,0xcc,0x07,0xfc]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} C_NGT_D32_MM

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
    c.f.s $f6, $f7
    c.un.s   $f6, $f7
    c.eq.s   $f6, $f7
    c.olt.s  $f6, $f7
    c.ult.s  $f6, $f7
    c.ole.s  $f6, $f7
    c.ule.s  $f6, $f7
    c.sf.s   $f6, $f7
    c.ngle.s $f6, $f7
    c.seq.s  $f6, $f7
    c.ngl.s  $f6, $f7
    c.lt.s   $f6, $f7
    c.nge.s  $f6, $f7
    c.le.s   $f6, $f7
    c.ngt.s  $f6, $f7
    c.sf.d   $f30, $f0
    c.f.d    $f12, $f14
    c.un.d   $f12, $f14
    c.eq.d   $f12, $f14
    c.ueq.d  $f12, $f14
    c.olt.d  $f12, $f14
    c.ult.d  $f12, $f14
    c.ole.d  $f12, $f14
    c.ule.d  $f12, $f14
    c.sf.d   $f12, $f14
    c.ngle.d $f12, $f14
    c.seq.d  $f12, $f14
    c.ngl.d  $f12, $f14
    c.lt.d   $f12, $f14
    c.nge.d  $f12, $f14
    c.le.d   $f12, $f14
    c.ngt.d  $f12, $f14
