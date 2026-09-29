# RUN: llvm-mc %s -triple=mipsel -show-encoding -mattr=micromips -show-inst \
# RUN: | FileCheck -check-prefix=CHECK-EL %s
# RUN: llvm-mc %s -triple=mips -show-encoding -mattr=micromips -show-inst \
# RUN: | FileCheck -check-prefix=CHECK-EB %s
# Check that the assembler can handle the documented syntax
# for load and store instructions.
#------------------------------------------------------------------------------
# Load and Store Instructions
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL: lb     $5, 8($4)                  # encoding: [0xa4,0x1c,0x08,0x00]
# CHECK-EL: lbu    $6, 8($4)                  # encoding: [0xc4,0x14,0x08,0x00]
# CHECK-EL: lh     $2, 8($4)                  # encoding: [0x44,0x3c,0x08,0x00]
# CHECK-EL: lhu    $4, 8($2)                  # encoding: [0x82,0x34,0x08,0x00]
# CHECK-EL: lw     $6, 4($5)                  # encoding: [0xc5,0xfc,0x04,0x00]
# CHECK-EL: lw     $6, 123($sp)               # encoding: [0xdd,0xfc,0x7b,0x00]
# CHECK-EL: sb     $5, 8($4)                  # encoding: [0xa4,0x18,0x08,0x00]
# CHECK-EL: sh     $2, 8($4)                  # encoding: [0x44,0x38,0x08,0x00]
# CHECK-EL: sw     $5, 4($6)                  # encoding: [0xa6,0xf8,0x04,0x00]
# CHECK-EL: sw     $5, 123($sp)               # encoding: [0xbd,0xf8,0x7b,0x00]
# CHECK-EL: sw     $3, 32($gp)                # encoding: [0x7c,0xf8,0x20,0x00]
# CHECK-EL: ll     $2, 8($4)                  # encoding: [0x44,0x60,0x08,0x30]
# CHECK-EL: sc     $2, 8($4)                  # encoding: [0x44,0x60,0x08,0xb0]
# CHECK-EL: lwu    $2, 8($4)                  # encoding: [0x44,0x60,0x08,0xe0]
# CHECK-EL: lwxs   $2, $3($4)                 # encoding: [0x64,0x00,0x18,0x11]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} LWXS_MM
# CHECK-EL: lwm32  $16, $17, 8($4)            # encoding: [0x44,0x20,0x08,0x50]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EL: lwm32  $16, $17, $18, $19, 8($4)  # encoding: [0x84,0x20,0x08,0x50]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EL: lwm32  $16, $17, $18, $19, $20, $21, $22, $23, $fp, 8($4)      # encoding: [0x24,0x21,0x08,0x50]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EL: lwm32  $16, $17, $18, $19, $ra, 8($4)                          # encoding: [0x84,0x22,0x08,0x50]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EL: lwm32  $16, $17, $18, $19, $20, $21, $22, $23, $fp, $ra, 8($4) # encoding: [0x24,0x23,0x08,0x50]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EL: lwm32  $16, $17, $18, $19, $20, $21, $22, $23, $fp, $ra, 8($4) # encoding: [0x24,0x23,0x08,0x50]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EL: swm32  $16, $17, 8($4)            # encoding: [0x44,0x20,0x08,0xd0]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} SWM32_MM
# CHECK-EL: swm32  $16, $17, $18, $19, 8($4)  # encoding: [0x84,0x20,0x08,0xd0]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} SWM32_MM
# CHECK-EL: lwm16  $16, $17, $ra, 8($sp)      # encoding: [0x12,0x45]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} LWM16_MM
# CHECK-EL: swm16  $16, $17, $ra, 8($sp)      # encoding: [0x52,0x45]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} SWM16_MM
# CHECK-EL: lwm16  $16, $17, $ra, 8($sp)      # encoding: [0x12,0x45]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} LWM16_MM
# CHECK-EL: lwm32  $16, $17, $ra, 64($sp)     # encoding: [0x5d,0x22,0x40,0x50]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EL: lwm32  $16, $17, $ra, 8($4)       # encoding: [0x44,0x22,0x08,0x50]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EL: lwm32  $16, $17, 8($sp)           # encoding: [0x5d,0x20,0x08,0x50]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EL: swm16  $16, $17, $ra, 8($sp)      # encoding: [0x52,0x45]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} SWM16_MM
# CHECK-EL: swm32  $16, $17, $ra, 64($sp)     # encoding: [0x5d,0x22,0x40,0xd0]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} SWM32_MM
# CHECK-EL: swm32  $16, $17, $ra, 8($4)       # encoding: [0x44,0x22,0x08,0xd0]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} SWM32_MM
# CHECK-EL: swm32  $16, $17, 8($sp)           # encoding: [0x5d,0x20,0x08,0xd0]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} SWM32_MM
# CHECK-EL: swp    $16, 8($4)                 # encoding: [0x04,0x22,0x08,0x90]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} SWP_MM
# CHECK-EL: lwp    $16, 8($4)                 # encoding: [0x04,0x22,0x08,0x10]
# CHECK-EL-NEXT:                              # <MCInst #{{[0-9]+}} LWP_MM
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB: lb     $5, 8($4)                 # encoding: [0x1c,0xa4,0x00,0x08]
# CHECK-EB: lbu    $6, 8($4)                 # encoding: [0x14,0xc4,0x00,0x08]
# CHECK-EB: lh     $2, 8($4)                 # encoding: [0x3c,0x44,0x00,0x08]
# CHECK-EB: lhu    $4, 8($2)                 # encoding: [0x34,0x82,0x00,0x08]
# CHECK-EB: lw     $6, 4($5)                 # encoding: [0xfc,0xc5,0x00,0x04]
# CHECK-EB: lw     $6, 123($sp)              # encoding: [0xfc,0xdd,0x00,0x7b]
# CHECK-EB: sb     $5, 8($4)                 # encoding: [0x18,0xa4,0x00,0x08]
# CHECK-EB: sh     $2, 8($4)                 # encoding: [0x38,0x44,0x00,0x08]
# CHECK-EB: sw     $5, 4($6)                 # encoding: [0xf8,0xa6,0x00,0x04]
# CHECK-EB: sw     $5, 123($sp)              # encoding: [0xf8,0xbd,0x00,0x7b]
# CHECK-EB: sw     $3, 32($gp)               # encoding: [0xf8,0x7c,0x00,0x20]
# CHECK-EB: ll     $2, 8($4)                 # encoding: [0x60,0x44,0x30,0x08]
# CHECK-EB: sc     $2, 8($4)                 # encoding: [0x60,0x44,0xb0,0x08]
# CHECK-EB: lwu    $2, 8($4)                 # encoding: [0x60,0x44,0xe0,0x08]
# CHECK-EB: lwxs   $2, $3($4)                # encoding: [0x00,0x64,0x11,0x18]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} LWXS_MM
# CHECK-EB: lwm32  $16, $17, 8($4)           # encoding: [0x20,0x44,0x50,0x08]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EB: lwm32  $16, $17, $18, $19, 8($4) # encoding: [0x20,0x84,0x50,0x08]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EB: lwm32  $16, $17, $18, $19, $20, $21, $22, $23, $fp, 8($4)      # encoding: [0x21,0x24,0x50,0x08]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EB: lwm32  $16, $17, $18, $19, $ra, 8($4)                          # encoding: [0x22,0x84,0x50,0x08]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EB: lwm32  $16, $17, $18, $19, $20, $21, $22, $23, $fp, $ra, 8($4) # encoding: [0x23,0x24,0x50,0x08]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EB: lwm32  $16, $17, $18, $19, $20, $21, $22, $23, $fp, $ra, 8($4) # encoding: [0x23,0x24,0x50,0x08]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EB: swm32  $16, $17, 8($4)           # encoding: [0x20,0x44,0xd0,0x08]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} SWM32_MM
# CHECK-EB: swm32  $16, $17, $18, $19, 8($4) # encoding: [0x20,0x84,0xd0,0x08]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} SWM32_MM
# CHECK-EB: lwm16  $16, $17, $ra, 8($sp)     # encoding: [0x45,0x12]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} LWM16_MM
# CHECK-EB: swm16  $16, $17, $ra, 8($sp)     # encoding: [0x45,0x52]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} SWM16_MM
# CHECK-EB: lwm16  $16, $17, $ra, 8($sp)     # encoding: [0x45,0x12]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} LWM16_MM
# CHECK-EB: lwm32  $16, $17, $ra, 64($sp)    # encoding: [0x22,0x5d,0x50,0x40]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EB: lwm32  $16, $17, $ra, 8($4)      # encoding: [0x22,0x44,0x50,0x08]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EB: lwm32  $16, $17, 8($sp)          # encoding: [0x20,0x5d,0x50,0x08]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} LWM32_MM
# CHECK-EB: swm16  $16, $17, $ra, 8($sp)     # encoding: [0x45,0x52]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} SWM16_MM
# CHECK-EB: swm32  $16, $17, $ra, 64($sp)    # encoding: [0x22,0x5d,0xd0,0x40]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} SWM32_MM
# CHECK-EB: swm32  $16, $17, $ra, 8($4)      # encoding: [0x22,0x44,0xd0,0x08]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} SWM32_MM
# CHECK-EB: swm32  $16, $17, 8($sp)          # encoding: [0x20,0x5d,0xd0,0x08]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} SWM32_MM
# CHECK-EB: swp    $16, 8($4)                # encoding: [0x22,0x04,0x90,0x08]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} SWP_MM
# CHECK-EB: lwp    $16, 8($4)                # encoding: [0x22,0x04,0x10,0x08]
# CHECK-EB-NEXT:                             # <MCInst #{{[0-9]+}} LWP_MM
     lb     $5, 8($4)
     lbu    $6, 8($4)
     lh     $2, 8($4)
     lhu    $4, 8($2)
     lw     $6, 4($5)
     lw     $6, 123($sp)
     sb     $5, 8($4)
     sh     $2, 8($4)
     sw     $5, 4($6)
     sw     $5, 123($sp)
     sw     $3, 32($gp)
     ll     $2, 8($4)
     sc     $2, 8($4)
     lwu    $2, 8($4)
     lwxs   $2, $3($4)
     lwm32  $16, $17, 8($4)
     lwm32  $16 - $19, 8($4)
     lwm32  $16-$23, $30, 8($4)
     lwm32  $16-$19, $31, 8($4)
     lwm32  $16-$23, $30, $31, 8($4)
     lwm32  $16-$23, $30 - $31, 8($4)
     swm32  $16, $17, 8($4)
     swm32  $16 - $19, 8($4)
     lwm16  $16, $17, $ra, 8($sp)
     swm16  $16, $17, $ra, 8($sp)
     lwm    $16, $17, $ra, 8($sp)
     lwm    $16, $17, $ra, 64($sp)
     lwm    $16, $17, $ra, 8($4)
     lwm    $16, $17, 8($sp)
     swm    $16, $17, $ra, 8($sp)
     swm    $16, $17, $ra, 64($sp)
     swm    $16, $17, $ra, 8($4)
     swm    $16, $17, 8($sp)
     swp    $16, 8($4)
     lwp    $16, 8($4)

