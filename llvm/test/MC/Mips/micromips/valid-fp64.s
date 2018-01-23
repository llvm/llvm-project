# RUN: llvm-mc -arch=mips -mcpu=mips32r3 -mattr=+micromips,+fp64 -show-encoding -show-inst %s \
# RUN: | FileCheck %s 

abs.s  $f0, $f12  # CHECK: abs.s  $f0, $f12  # encoding: [0x54,0x0c,0x03,0x7b]
                  # CHECK-NEXT:              # <MCInst #{{[0-9]+}} FABS_S_MM
abs.d  $f0, $f12  # CHECK: abs.d  $f0, $f12  # encoding: [0x54,0x0c,0x23,0x7b]
                  # CHECK-NEXT:              # <MCInst #{{[0-9]+}} FABS_D64_MM
sqrt.s $f0, $f12  # CHECK: sqrt.s $f0, $f12  # encoding: [0x54,0x0c,0x0a,0x3b]
                  # CHECK-NEXT:              # <MCInst #{{[0-9]+}} FSQRT_S_MM
sqrt.d $f0, $f12  # CHECK: sqrt.d $f0, $f12  # encoding: [0x54,0x0c,0x4a,0x3b]
                  # CHECK-NEXT:              # <MCInst #{{[0-9]+}} FSQRT_D64_MM
