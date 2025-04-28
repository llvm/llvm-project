# RUN: llvm-mc -triple=mips -mcpu=mips32 -show-encoding %s | FileCheck %s

# Check that parseMemOperand handles expressions such as <int>, (<int>),
# <expr>, <expr> op <expr>, (<expr>) op (<expr>).

    .global __start
    .ent    __start
__start:
    lw $31, ($29)                       # CHECK: lw  $ra, 0($sp)      # encoding: [0x8f,0xbf,0x00,0x00]
    lw $31, 0($29)                      # CHECK: lw  $ra, 0($sp)      # encoding: [0x8f,0xbf,0x00,0x00]
    lw $31, (8)($29)                    # CHECK: lw  $ra, 8($sp)      # encoding: [0x8f,0xbf,0x00,0x08]
    lw $31, 3 + (4 * 8)($29)            # CHECK: lw  $ra, 35($sp)     # encoding: [0x8f,0xbf,0x00,0x23]
    lw $31, (8 + 8)($29)                # CHECK: lw  $ra, 16($sp)     # encoding: [0x8f,0xbf,0x00,0x10]
    lw $31, (8 << 4)($29)               # CHECK: lw  $ra, 128($sp)    # encoding: [0x8f,0xbf,0x00,0x80]
    lw $31, (32768 >> 2)($29)           # CHECK: lw  $ra, 8192($sp)   # encoding: [0x8f,0xbf,0x20,0x00]
    lw $31, 32768 >> 2($29)             # CHECK: lw  $ra, 8192($sp)   # encoding: [0x8f,0xbf,0x20,0x00]
    lw $31, 2 << 3($29)                 # CHECK: lw  $ra, 16($sp)     # encoding: [0x8f,0xbf,0x00,0x10]
    lw $31, (2 << 3)($29)               # CHECK: lw  $ra, 16($sp)     # encoding: [0x8f,0xbf,0x00,0x10]
    lw $31, 4 - (4 * 8)($29)            # CHECK: lw  $ra, -28($sp)    # encoding: [0x8f,0xbf,0xff,0xe4]
    lw $31, 4 | 8 ($29)                 # CHECK: lw  $ra, 12($sp)     # encoding: [0x8f,0xbf,0x00,0x0c]
    lw $31, 4 || 8 ($29)                # CHECK: lw  $ra, 1($sp)      # encoding: [0x8f,0xbf,0x00,0x01]
    lw $31, 8 & 8 ($29)                 # CHECK: lw  $ra, 8($sp)      # encoding: [0x8f,0xbf,0x00,0x08]
    lw $31, (8 * 4) ^ (8 * 31)($29)     # CHECK: lw  $ra, 216($sp)    # encoding: [0x8f,0xbf,0x00,0xd8]
    lw $31, (8 * 4) / (8 * 31)($29)     # CHECK: lw  $ra, 0($sp)      # encoding: [0x8f,0xbf,0x00,0x00]
    lw $31, (8 * 4) % (8 * 31)($29)     # CHECK: lw  $ra, 32($sp)     # encoding: [0x8f,0xbf,0x00,0x20]
    lw $31, (8 * 4) % (8)($29)          # CHECK: lw  $ra, 0($sp)      # encoding: [0x8f,0xbf,0x00,0x00]
    lw $31, (8 * 4) + (8 * 31) ($29)    # CHECK: lw  $ra, 280($sp)    # encoding: [0x8f,0xbf,0x01,0x18]
    lw $31, (8*4) + (8*31) + (8*32 + __start) ($29) # CHECK:  lui $ra, %hi((248+((8*32)+__start))+32) # encoding: [0x3c,0x1f,A,A]
                                                    # CHECK:                                #   fixup A - offset: 0, value: %hi((248+((8*32)+__start))+32), kind: fixup_Mips_HI16
                                                    # CHECK:  addu  $ra, $ra, $sp           # encoding: [0x03,0xfd,0xf8,0x21]
                                                    # CHECK:  lw  $ra, %lo((248+((8*32)+__start))+32)($ra) # encoding: [0x8f,0xff,A,A]
                                                    # CHECK:                                #   fixup A - offset: 0, value: %lo((248+((8*32)+__start))+32), kind: fixup_Mips_LO16
    .end __start
