# RUN: llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips,fp64 \
# RUN:   -show-inst -mcpu=mips32r2 | FileCheck -check-prefix=CHECK-EL %s
# RUN: llvm-mc %s -triple=mips -show-encoding -mattr=micromips,fp64 \
# RUN:   -show-inst -mcpu=mips32r2 | FileCheck -check-prefix=CHECK-EB %s
# Check that the assembler can handle the documented syntax
# for fpu instructions
#------------------------------------------------------------------------------
# FPU Instructions
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL: luxc1      $f2, $4($6)      # encoding: [0x86,0x54,0x48,0x11]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} LUXC1_MM
# CHECK-EL: suxc1      $f2, $4($6)      # encoding: [0x86,0x54,0x88,0x11]
# CHECK-EL-NEXT:                        # <MCInst #{{[0-9]+}} SUXC1_MM
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB: luxc1 $f2, $4($6)           # encoding: [0x54,0x86,0x11,0x48]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} LUXC1_MM
# CHECK-EB: suxc1 $f2, $4($6)           # encoding: [0x54,0x86,0x11,0x88]
# CHECK-EB-NEXT:                        # <MCInst #{{[0-9]+}} SUXC1_MM

    luxc1      $f2, $4($6)
    suxc1      $f2, $4($6)

