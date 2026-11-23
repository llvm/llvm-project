# RUN: llvm-mc %s -triple=mips64el-unknown-linux -mcpu=r5900 -show-encoding \
# RUN:   | FileCheck %s

# Test R5900 FPU instructions (single-precision only, no double-precision support)
# The R5900 has 32x32-bit FPU registers and a 32-bit accumulator (ACC)

#------------------------------------------------------------------------------
# FP Arithmetic Instructions
#------------------------------------------------------------------------------

# CHECK: abs.s   $f6, $f7           # encoding: [0x85,0x39,0x00,0x46]
abs.s   $f6, $f7

# CHECK: add.s   $f9, $f6, $f7      # encoding: [0x40,0x32,0x07,0x46]
add.s   $f9, $f6, $f7

# CHECK: sub.s   $f9, $f6, $f7      # encoding: [0x41,0x32,0x07,0x46]
sub.s   $f9, $f6, $f7

# CHECK: mul.s   $f9, $f6, $f7      # encoding: [0x42,0x32,0x07,0x46]
mul.s   $f9, $f6, $f7

# CHECK: div.s   $f9, $f6, $f7      # encoding: [0x43,0x32,0x07,0x46]
div.s   $f9, $f6, $f7

# CHECK: sqrt.s  $f6, $f7           # encoding: [0x84,0x39,0x00,0x46]
sqrt.s  $f6, $f7

# CHECK: neg.s   $f6, $f7           # encoding: [0x87,0x39,0x00,0x46]
neg.s   $f6, $f7

# CHECK: mov.s   $f6, $f7           # encoding: [0x86,0x39,0x00,0x46]
mov.s   $f6, $f7

#------------------------------------------------------------------------------
# FP Conversion Instructions
#------------------------------------------------------------------------------

# CHECK: cvt.s.w $f6, $f7           # encoding: [0xa0,0x39,0x80,0x46]
cvt.s.w $f6, $f7

# CHECK: cvt.w.s $f6, $f7           # encoding: [0xa4,0x39,0x00,0x46]
cvt.w.s $f6, $f7

#------------------------------------------------------------------------------
# FP Compare Instructions (R5900 supports only 4 comparisons: F, EQ, OLT, OLE)
#------------------------------------------------------------------------------

# CHECK: c.f.s   $f6, $f7           # encoding: [0x30,0x30,0x07,0x46]
c.f.s   $f6, $f7

# CHECK: c.eq.s  $f6, $f7           # encoding: [0x32,0x30,0x07,0x46]
c.eq.s  $f6, $f7

# CHECK: c.olt.s $f6, $f7           # encoding: [0x34,0x30,0x07,0x46]
c.olt.s $f6, $f7

# CHECK: c.ole.s $f6, $f7           # encoding: [0x36,0x30,0x07,0x46]
c.ole.s $f6, $f7

#------------------------------------------------------------------------------
# FP Branch Instructions
#------------------------------------------------------------------------------

# CHECK: bc1f    8                  # encoding: [0x02,0x00,0x00,0x45]
bc1f    8

# CHECK: bc1fl   8                  # encoding: [0x02,0x00,0x02,0x45]
bc1fl   8

# CHECK: bc1t    8                  # encoding: [0x02,0x00,0x01,0x45]
bc1t    8

# CHECK: bc1tl   8                  # encoding: [0x02,0x00,0x03,0x45]
bc1tl   8

#------------------------------------------------------------------------------
# FP Load/Store Instructions
#------------------------------------------------------------------------------

# CHECK: lwc1    $f6, 4($5)         # encoding: [0x04,0x00,0xa6,0xc4]
lwc1    $f6, 4($5)

# CHECK: swc1    $f6, 4($5)         # encoding: [0x04,0x00,0xa6,0xe4]
swc1    $f6, 4($5)

#------------------------------------------------------------------------------
# FP Move Instructions (between GPR and FPR)
#------------------------------------------------------------------------------

# CHECK: mfc1    $6, $f7            # encoding: [0x00,0x38,0x06,0x44]
mfc1    $6, $f7

# CHECK: mtc1    $6, $f7            # encoding: [0x00,0x38,0x86,0x44]
mtc1    $6, $f7

#------------------------------------------------------------------------------
# FP Control Register Move Instructions
#------------------------------------------------------------------------------

# CHECK: cfc1    $6, $0             # encoding: [0x00,0x00,0x46,0x44]
cfc1    $6, $0

# CHECK: ctc1    $6, $31            # encoding: [0x00,0xf8,0xc6,0x44]
ctc1    $6, $31

# TODO: R5900 FP Accumulator Instructions
# #------------------------------------------------------------------------------
# # R5900-Specific: FP Accumulator Instructions
# #------------------------------------------------------------------------------
#
# # DISABLED: adda.s  $f1, $f2           # encoding: [0x18,0x08,0x02,0x46]
# adda.s  $f1, $f2
#
# # DISABLED: suba.s  $f3, $f4           # encoding: [0x19,0x18,0x04,0x46]
# suba.s  $f3, $f4
#
# # DISABLED: mula.s  $f5, $f6           # encoding: [0x1a,0x28,0x06,0x46]
# mula.s  $f5, $f6
#
# # DISABLED: madda.s $f7, $f8           # encoding: [0x1e,0x38,0x08,0x46]
# madda.s $f7, $f8
#
# # DISABLED: msuba.s $f9, $f10          # encoding: [0x1f,0x48,0x0a,0x46]
# msuba.s $f9, $f10
#
# # DISABLED: madd.s  $f0, $f1, $f2      # encoding: [0x1c,0x08,0x02,0x46]
# madd.s  $f0, $f1, $f2
#
# # DISABLED: msub.s  $f3, $f4, $f5      # encoding: [0xdd,0x20,0x05,0x46]
# msub.s  $f3, $f4, $f5

# TODO: R5900-Specific FPU Instructions (max.s, min.s, rsqrt.s)
# #------------------------------------------------------------------------------
# # R5900-Specific: FP Min/Max Instructions
# #------------------------------------------------------------------------------
#
# # DISABLED: max.s   $f0, $f1, $f2      # encoding: [0x28,0x08,0x02,0x46]
# max.s   $f0, $f1, $f2
#
# # DISABLED: min.s   $f3, $f4, $f5      # encoding: [0xe9,0x20,0x05,0x46]
# min.s   $f3, $f4, $f5
#
# #------------------------------------------------------------------------------
# # R5900-Specific: RSQRT.S (3-operand: fd = fs / sqrt(ft))
# # Note: differs from MIPS IV 2-operand rsqrt.s
# #------------------------------------------------------------------------------
#
# # DISABLED: rsqrt.s $f6, $f7, $f8      # encoding: [0x96,0x39,0x08,0x46]
# rsqrt.s $f6, $f7, $f8
