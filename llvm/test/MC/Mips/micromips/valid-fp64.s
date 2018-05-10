# RUN: llvm-mc -arch=mips -mcpu=mips32r3 -mattr=+micromips,+fp64 -show-encoding -show-inst %s \
# RUN: | FileCheck %s 

abs.d   $f0, $f12     # CHECK: abs.d  $f0, $f12      # encoding: [0x54,0x0c,0x23,0x7b]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FABS_D64_MM
abs.s   $f0, $f12     # CHECK: abs.s  $f0, $f12      # encoding: [0x54,0x0c,0x03,0x7b]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FABS_S_MM
add.d   $f0, $f2, $f4 # CHECK: add.d   $f0, $f2, $f4 # encoding: [0x54,0x82,0x01,0x30]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FADD_D64_MM
cvt.d.s $f0, $f2      # CHECK: cvt.d.s $f0, $f2      # encoding: [0x54,0x02,0x13,0x7b]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} CVT_D64_S_MM
cvt.d.w $f0, $f2      # CHECK: cvt.d.w $f0, $f2      # encoding: [0x54,0x02,0x33,0x7b]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} CVT_D64_W_MM
cvt.s.d $f0, $f2      # CHECK: cvt.s.d $f0, $f2      # encoding: [0x54,0x02,0x1b,0x7b]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} CVT_S_D64_MM
cvt.w.d $f0, $f2      # CHECK: cvt.w.d $f0, $f2      # encoding: [0x54,0x02,0x49,0x3b]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} CVT_W_D64_MM
cvt.l.s $f4, $f2      # CHECK:  cvt.l.s $f4, $f2     # encoding: [0x54,0x82,0x01,0x3b]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} CVT_L_S_MM
cvt.l.d $f4, $f2      # CHECK:  cvt.l.d $f4, $f2     # encoding: [0x54,0x82,0x41,0x3b]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} CVT_L_D64_MM
div.d   $f0, $f2, $f4 # CHECK: div.d   $f0, $f2, $f4 # encoding: [0x54,0x82,0x01,0xf0]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FDIV_D64_MM
mfhc1   $4, $f0       # CHECK: mfhc1   $4, $f0       # encoding: [0x54,0x80,0x30,0x3b]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} MFHC1_D64_MM
mov.d   $f0, $f2      # CHECK: mov.d   $f0, $f2      # encoding: [0x54,0x02,0x20,0x7b]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FMOV_D64_MM
mthc1   $4, $f0       # CHECK: mthc1   $4, $f0       # encoding: [0x54,0x80,0x38,0x3b]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} MTHC1_D64_MM
mul.d   $f0, $f2, $f4 # CHECK: mul.d $f0, $f2, $f4   # encoding: [0x54,0x82,0x01,0xb0]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FMUL_D64_MM
neg.d   $f0, $f2      # CHECK: neg.d   $f0, $f2      # encoding: [0x54,0x02,0x2b,0x7b]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FNEG_D64_MM
sqrt.d  $f0, $f12     # CHECK: sqrt.d $f0, $f12      # encoding: [0x54,0x0c,0x4a,0x3b]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FSQRT_D64_MM
sqrt.s  $f0, $f12     # CHECK: sqrt.s $f0, $f12      # encoding: [0x54,0x0c,0x0a,0x3b]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FSQRT_S_MM
sub.d   $f0, $f2, $f4 # CHECK: sub.d $f0, $f2, $f4   # encoding: [0x54,0x82,0x01,0x70]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FSUB_D64_MM
