# RUN: llvm-mc %s -triple=mipsel -show-encoding -mcpu=mips32r2 -mattr=micromips \
# RUN: | FileCheck -check-prefix=CHECK-EL %s
# RUN: llvm-mc %s -triple=mips -show-encoding -mcpu=mips32r2 -mattr=micromips \
# RUN: | FileCheck -check-prefix=CHECK-EB %s
# Check that the assembler can handle the documented syntax
# for control instructions.
#------------------------------------------------------------------------------
# microMIPS Control Instructions
#------------------------------------------------------------------------------
# Little endian
#------------------------------------------------------------------------------
# CHECK-EL:    sdbbp                      # encoding: [0x00,0x00,0x7c,0xdb]
# CHECK-EL:    sdbbp 34                   # encoding: [0x22,0x00,0x7c,0xdb]
# CHECK-EL-NOT:    .set push
# CHECK-EL-NOT:    .set mips32r2
# CHECK-EL:    rdhwr $5, $29              # encoding: [0xbd,0x00,0x3c,0x6b]
# CHECK-EL-NOT:    .set pop
# CHECK-EL:    cache 1, 8($5)             # encoding: [0x25,0x20,0x08,0x60]
# CHECK-EL:    pref 1, 8($5)              # encoding: [0x25,0x60,0x08,0x20]
# CHECK-EL:    ssnop                      # encoding: [0x00,0x00,0x00,0x08]
# CHECK-EL:    ehb                        # encoding: [0x00,0x00,0x00,0x18]
# CHECK-EL:    pause                      # encoding: [0x00,0x00,0x00,0x28]
# CHECK-EL:    break                      # encoding: [0x00,0x00,0x07,0x00]
# CHECK-EL:    break 7                    # encoding: [0x07,0x00,0x07,0x00]
# CHECK-EL:    break 7, 5                 # encoding: [0x07,0x00,0x47,0x01]
# CHECK-EL:    syscall                    # encoding: [0x00,0x00,0x7c,0x8b]
# CHECK-EL:    syscall 396                # encoding: [0x8c,0x01,0x7c,0x8b]
# CHECK-EL:    eret                       # encoding: [0x00,0x00,0x7c,0xf3]
# CHECK-EL:    deret                      # encoding: [0x00,0x00,0x7c,0xe3]
# CHECK-EL:    di                         # encoding: [0x00,0x00,0x7c,0x47]
# CHECK-EL:    di                         # encoding: [0x00,0x00,0x7c,0x47]
# CHECK-EL:    di  $10                    # encoding: [0x0a,0x00,0x7c,0x47]
# CHECK-EL:    ei                         # encoding: [0x00,0x00,0x7c,0x57]
# CHECK-EL:    ei                         # encoding: [0x00,0x00,0x7c,0x57]
# CHECK-EL:    ei  $10                    # encoding: [0x0a,0x00,0x7c,0x57]
# CHECK-EL:    wait                       # encoding: [0x00,0x00,0x7c,0x93]
# CHECK-EL:    wait 17                    # encoding: [0x11,0x00,0x7c,0x93]
# CHECK-EL:    tlbp                       # encoding: [0x00,0x00,0x7c,0x03]
# CHECK-EL:    tlbr                       # encoding: [0x00,0x00,0x7c,0x13]
# CHECK-EL:    tlbwi                      # encoding: [0x00,0x00,0x7c,0x23]
# CHECK-EL:    tlbwr                      # encoding: [0x00,0x00,0x7c,0x33]
# CHECK-EL:    prefx 1, $3($5)            # encoding: [0x65,0x54,0xa0,0x09]
#------------------------------------------------------------------------------
# Big endian
#------------------------------------------------------------------------------
# CHECK-EB:   sdbbp                       # encoding: [0x00,0x00,0xdb,0x7c]
# CHECK-EB:   sdbbp 34                    # encoding: [0x00,0x22,0xdb,0x7c]
# CHECK-EB-NOT:    .set push
# CHECK-EB-NOT:    .set mips32r2
# CHECK-EB:   rdhwr $5, $29               # encoding: [0x00,0xbd,0x6b,0x3c]
# CHECK-EB-NOT:    .set pop
# CHECK-EB:   cache 1, 8($5)              # encoding: [0x20,0x25,0x60,0x08]
# CHECK-EB:   pref 1, 8($5)               # encoding: [0x60,0x25,0x20,0x08]
# CHECK-EB:   ssnop                       # encoding: [0x00,0x00,0x08,0x00]
# CHECK-EB:   ehb                         # encoding: [0x00,0x00,0x18,0x00]
# CHECK-EB:   pause                       # encoding: [0x00,0x00,0x28,0x00]
# CHECK-EB:   break                       # encoding: [0x00,0x00,0x00,0x07]
# CHECK-EB:   break 7                     # encoding: [0x00,0x07,0x00,0x07]
# CHECK-EB:   break 7, 5                  # encoding: [0x00,0x07,0x01,0x47]
# CHECK-EB:   syscall                     # encoding: [0x00,0x00,0x8b,0x7c]
# CHECK-EB:   syscall 396                 # encoding: [0x01,0x8c,0x8b,0x7c]
# CHECK-EB:   eret                        # encoding: [0x00,0x00,0xf3,0x7c]
# CHECK-EB:   deret                       # encoding: [0x00,0x00,0xe3,0x7c]
# CHECK-EB:   di                          # encoding: [0x00,0x00,0x47,0x7c]
# CHECK-EB:   di                          # encoding: [0x00,0x00,0x47,0x7c]
# CHECK-EB:   di  $10                     # encoding: [0x00,0x0a,0x47,0x7c]
# CHECK-EB:   ei                          # encoding: [0x00,0x00,0x57,0x7c]
# CHECK-EB:   ei                          # encoding: [0x00,0x00,0x57,0x7c]
# CHECK-EB:   ei  $10                     # encoding: [0x00,0x0a,0x57,0x7c]
# CHECK-EB:   wait                        # encoding: [0x00,0x00,0x93,0x7c]
# CHECK-EB:   wait 17                     # encoding: [0x00,0x11,0x93,0x7c]
# CHECK-EB:   tlbp                        # encoding: [0x00,0x00,0x03,0x7c]
# CHECK-EB:   tlbr                        # encoding: [0x00,0x00,0x13,0x7c]
# CHECK-EB:   tlbwi                       # encoding: [0x00,0x00,0x23,0x7c]
# CHECK-EB:   tlbwr                       # encoding: [0x00,0x00,0x33,0x7c]
# CHECK-EB:   prefx 1, $3($5)             # encoding: [0x54,0x65,0x09,0xa0]

    sdbbp
    sdbbp 34
    rdhwr $5, $29
    cache 1, 8($5)
    pref 1, 8($5)
    ssnop
    ehb
    pause
    break
    break 7
    break 7,5
    syscall
    syscall 0x18c
    eret
    deret
    di
    di $0
    di $10
    ei
    ei $0
    ei $10
    wait
    wait 17
    tlbp
    tlbr
    tlbwi
    tlbwr
    prefx 1, $3($5)

