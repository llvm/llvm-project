# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 | FileCheck %s
# Check that the assembler can handle the documented syntax
# for jumps and branches.
# CHECK: .section __TEXT,__text,regular,pure_instructions
#------------------------------------------------------------------------------
# Branch instructions
#------------------------------------------------------------------------------
# CHECK:   b 1332                 # encoding: [0x34,0x05,0x00,0x10]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   bc1f 1332              # encoding: [0x34,0x05,0x00,0x45]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   bc1t 1332              # encoding: [0x34,0x05,0x01,0x45]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   beq $9, $6, 1332       # encoding: [0x34,0x05,0x26,0x11]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   bgez $6, 1332          # encoding: [0x34,0x05,0xc1,0x04]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   bgezal $6, 1332        # encoding: [0x34,0x05,0xd1,0x04]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   bgtz $6, 1332          # encoding: [0x34,0x05,0xc0,0x1c]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   blez $6, 1332          # encoding: [0x34,0x05,0xc0,0x18]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   bne $9, $6, 1332       # encoding: [0x34,0x05,0x26,0x15]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   bal     1332           # encoding: [0x34,0x05,0x00,0x04]
# CHECK:   nop                    # encoding: [0x00,0x00,0x00,0x00]
         b 1332
         nop
         bc1f 1332
         nop
         bc1t 1332
         nop
         beq $9,$6,1332
         nop
         bgez $6,1332
         nop
         bgezal $6,1332
         nop
         bgtz $6,1332
         nop
         blez $6,1332
         nop
         bne $9,$6,1332
         nop
         bal 1332
         nop

end_of_code:
#------------------------------------------------------------------------------
# Jump instructions
#------------------------------------------------------------------------------
# CHECK:   j 1328               # encoding: [0x30,0x05,0x00,0x08]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   jal 1328             # encoding: [0x30,0x05,0x00,0x0c]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   jalr $6              # encoding: [0x09,0xf8,0xc0,0x00]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   jr $7                # encoding: [0x08,0x00,0xe0,0x00]
# CHECK:   nop                  # encoding: [0x00,0x00,0x00,0x00]
# CHECK:   jr $7                # encoding: [0x08,0x00,0xe0,0x00]


   j 1328
   nop
   jal 1328
   nop
   jalr $6
   nop
   jr $7
   nop
   j $7
