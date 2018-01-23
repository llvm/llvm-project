# RUN: llvm-mc -arch=mips -mcpu=mips32r3 -mattr=+fp64 -show-encoding -show-inst %s | \
# RUN: FileCheck %s

abs.s $f0, $f12   # CHECK: abs.s  $f0, $f12  # encoding: [0x46,0x00,0x60,0x05]
                  # CHECK-NEXT:              # <MCInst #{{[0-9]+}} FABS_S
abs.d  $f0, $f12  # CHECK: abs.d  $f0, $f12  # encoding: [0x46,0x20,0x60,0x05]
                  # CHECK-NEXT:              # <MCInst #{{[0-9]+}} FABS_D64
sqrt.s  $f0, $f12 # CHECK: sqrt.s $f0, $f12  # encoding: [0x46,0x00,0x60,0x04]
                  # CHECK-NEXT:              # <MCInst #{{[0-9]+}} FSQRT_S
sqrt.d  $f0, $f12 # CHECK: sqrt.d $f0, $f12  # encoding: [0x46,0x20,0x60,0x04]
                  # CHECK-NEXT:              # <MCInst #{{[0-9]+}} FSQRT_D64
