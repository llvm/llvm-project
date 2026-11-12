# RUN: llvm-mc %s -triple=mipsel -show-encoding -show-inst -mcpu=mips32r6 \
# RUN:   -mattr=+micromips,+eva | FileCheck -check-prefix=CHECK-EL %s
# RUN: llvm-mc %s -triple=mips -show-encoding -show-inst -mcpu=mips32r6 \
# RUN:   -mattr=+micromips,+eva | FileCheck -check-prefix=CHECK-EB %s
# Check that the assembler can handle the documented syntax
# for EVA instructions.
#------------------------------------------------------------------------------
# microMIPS EVA Instructions
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL:    prefe 1, 8($5)             # encoding: [0x25,0x60,0x08,0xa4]
# CHECK-EL-NEXT:                          # <MCInst #{{[0-9]+}} PREFE_MM
# CHECK-EL:    cachee 1, 8($5)            # encoding: [0x25,0x60,0x08,0xa6]
# CHECK-EL-NEXT:                          # <MCInst #{{[0-9]+}} CACHEE_MM
# CHECK-EL:    lle $2, 8($4)              # encoding: [0x44,0x60,0x08,0x6c]
# CHECK-EL-NEXT:                          # <MCInst #{{[0-9]+}} LLE_MM
# CHECK-EL:    sce $2, 8($4)              # encoding: [0x44,0x60,0x08,0xac]
# CHECK-EL-NEXT:                          # <MCInst #{{[0-9]+}} SCE_MM
# CHECK-EL: lhue   $4, 8($2)              # encoding: [0x82,0x60,0x08,0x62]
# CHECK-EL-NEXT:                          # <MCInst #{{[0-9]+}} LHuE_M
# CHECK-EL: lbe    $4, 8($2)              # encoding: [0x82,0x60,0x08,0x68]
# CHECK-EL-NEXT:                          # <MCInst #{{[0-9]+}} LBE_MM
# CHECK-EL: lbue   $4, 8($2)              # encoding: [0x82,0x60,0x08,0x60]
# CHECK-EL-NEXT:                          # <MCInst #{{[0-9]+}} LBuE_MM
# CHECK-EL: lhe    $4, 8($2)              # encoding: [0x82,0x60,0x08,0x6a]
# CHECK-EL-NEXT:                          # <MCInst #{{[0-9]+}} LHE_MM
# CHECK-EL: lwe    $4, 8($2)              # encoding: [0x82,0x60,0x08,0x6e]
# CHECK-EL-NEXT:                          # <MCInst #{{[0-9]+}} LWE_MM
# CHECK-EL: sbe    $5, 8($4)              # encoding: [0xa4,0x60,0x08,0xa8]
# CHECK-EL-NEXT:                          # <MCInst #{{[0-9]+}} SBE_MM
# CHECK-EL: she    $5, 8($4)              # encoding: [0xa4,0x60,0x08,0xaa]
# CHECK-EL-NEXT:                          # <MCInst #{{[0-9]+}} SHE_MM
# CHECK-EL: swe    $5, 8($4)              # encoding: [0xa4,0x60,0x08,0xae]
# CHECK-EL-NEXT:                          # <MCInst #{{[0-9]+}} SWE_MM
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB:   prefe 1, 8($5)              # encoding: [0x60,0x25,0xa4,0x08]
# CHECK-EB-NEXT:                          # <MCInst #{{[0-9]+}} PREFE_MM
# CHECK-EB:   cachee 1, 8($5)             # encoding: [0x60,0x25,0xa6,0x08]
# CHECK-EB-NEXT:                          # <MCInst #{{[0-9]+}} CACHEE_MM
# CHECK-EB:   lle $2, 8($4)               # encoding: [0x60,0x44,0x6c,0x08]
# CHECK-EB-NEXT:                          # <MCInst #{{[0-9]+}} LLE_MM
# CHECK-EB:   sce $2, 8($4)               # encoding: [0x60,0x44,0xac,0x08]
# CHECK-EB-NEXT:                          # <MCInst #{{[0-9]+}} SCE_MM
# CHECK-EB: lhue   $4, 8($2)              # encoding: [0x60,0x82,0x62,0x08]
# CHECK-EB-NEXT:                          # <MCInst #{{[0-9]+}} LHuE_MM
# CHECK-EB: lbe    $4, 8($2)              # encoding: [0x60,0x82,0x68,0x08]
# CHECK-EB-NEXT:                          # <MCInst #{{[0-9]+}} LBE_MM
# CHECK-EB: lbue   $4, 8($2)              # encoding: [0x60,0x82,0x60,0x08]
# CHECK-EB-NEXT:                          # <MCInst #{{[0-9]+}} LBuE_MM
# CHECK-EB: lhe    $4, 8($2)              # encoding: [0x60,0x82,0x6a,0x08]
# CHECK-EB-NEXT:                          # <MCInst #{{[0-9]+}} LHE_MM
# CHECK-EB: lwe    $4, 8($2)              # encoding: [0x60,0x82,0x6e,0x08]
# CHECK-EB-NEXT:                          # <MCInst #{{[0-9]+}} LWE_MM
# CHECK-EB: sbe    $5, 8($4)              # encoding: [0x60,0xa4,0xa8,0x08]
# CHECK-EB-NEXT:                          # <MCInst #{{[0-9]+}} SBE_MM
# CHECK-EB: she    $5, 8($4)              # encoding: [0x60,0xa4,0xaa,0x08]
# CHECK-EB-NEXT:                          # <MCInst #{{[0-9]+}} SHE_MM
# CHECK-EB: swe    $5, 8($4)              # encoding: [0x60,0xa4,0xae,0x08]
# CHECK-EB-NEXT:                          # <MCInst #{{[0-9]+}} SWE_MM

    prefe 1, 8($5)
    cachee 1, 8($5)
    lle $2, 8($4)
    sce $2, 8($4)
    lhue   $4, 8($2)
    lbe    $4, 8($2)
    lbue   $4, 8($2)
    lhe    $4, 8($2)
    lwe    $4, 8($2)
    sbe    $5, 8($4)
    she    $5, 8($4)
    swe    $5, 8($4)

