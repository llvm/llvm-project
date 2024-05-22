# RUN: llvm-mc -triple=mips -mcpu=mips32r5 -mattr=+fp64 -show-encoding -show-inst %s | \
# RUN: FileCheck %s

abs.d  $f0, $f12      # CHECK: abs.d  $f0, $f12      # encoding: [0x46,0x20,0x60,0x05]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FABS_D64
abs.s $f0, $f12       # CHECK: abs.s  $f0, $f12      # encoding: [0x46,0x00,0x60,0x05]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FABS_S
add.d   $f0, $f2, $f4 # CHECK: add.d   $f0, $f2, $f4 # encoding: [0x46,0x24,0x10,0x00]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FADD_D64
cvt.d.s $f0, $f2      # CHECK: cvt.d.s $f0, $f2      # encoding: [0x46,0x00,0x10,0x21]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} CVT_D64_S
cvt.d.w $f0, $f2      # CHECK: cvt.d.w $f0, $f2      # encoding: [0x46,0x80,0x10,0x21]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} CVT_D64_W
cvt.s.d $f0, $f2      # CHECK: cvt.s.d $f0, $f2      # encoding: [0x46,0x20,0x10,0x20]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} CVT_S_D64
cvt.w.d $f0, $f2      # CHECK: cvt.w.d $f0, $f2      # encoding: [0x46,0x20,0x10,0x24]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} CVT_W_D64
div.d   $f0, $f2, $f4 # CHECK: div.d   $f0, $f2, $f4 # encoding: [0x46,0x24,0x10,0x03]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FDIV_D64
mfhc1   $4, $f0       # CHECK: mfhc1   $4, $f0       # encoding: [0x44,0x64,0x00,0x00]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} MFHC1_D64
mov.d   $f0, $f2      # CHECK: mov.d   $f0, $f2      # encoding: [0x46,0x20,0x10,0x06]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FMOV_D64
mthc1   $4, $f0       # CHECK: mthc1   $4, $f0       # encoding: [0x44,0xe4,0x00,0x00]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} MTHC1_D64
mul.d   $f0, $f2, $f4 # CHECK: mul.d $f0, $f2, $f4   # encoding: [0x46,0x24,0x10,0x02]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FMUL_D64
neg.d   $f0, $f2      # CHECK: neg.d   $f0, $f2      # encoding: [0x46,0x20,0x10,0x07]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FNEG_D64
sqrt.d  $f0, $f12     # CHECK: sqrt.d $f0, $f12      # encoding: [0x46,0x20,0x60,0x04]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FSQRT_D64
sqrt.s  $f0, $f12     # CHECK: sqrt.s $f0, $f12      # encoding: [0x46,0x00,0x60,0x04]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FSQRT_S
sub.d   $f0, $f2, $f4 # CHECK: sub.d $f0, $f2, $f4   # encoding: [0x46,0x24,0x10,0x01]
                      # CHECK-NEXT:                  # <MCInst #{{[0-9]+}} FSUB_D64
