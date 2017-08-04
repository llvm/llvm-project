# RUN: llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips3 | \
# RUN:   FileCheck %s

.option pic2
dla $5, symbol        # CHECK: ld $5, %got_disp(symbol)($gp)   # encoding: [0xdf,0x85,A,A]
                      # CHECK:                                 #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
dla $5, symbol($6)    # CHECK: ld $5, %got_disp(symbol)($gp)   # encoding: [0xdf,0x85,A,A]
                      # CHECK:                                 #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
                      # CHECK: daddu $5, $5, $6                # encoding: [0x00,0xa6,0x28,0x2d]
dla $6, symbol($6)    # CHECK: ld $1, %got_disp(symbol)($gp)   # encoding: [0xdf,0x81,A,A]
                      # CHECK:                                 #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
                      # CHECK: daddu $6, $1, $6                # encoding: [0x00,0x26,0x30,0x2d]
dla $5, symbol+8      # CHECK: ld $5, %got_disp(symbol)($gp)   # encoding: [0xdf,0x85,A,A]
                      # CHECK:                                 #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
                      # CHECK: daddiu $5, $5, 8                # encoding: [0x64,0xa5,0x00,0x08]
dla $5, symbol+8($6)  # CHECK: ld $5, %got_disp(symbol)($gp)   # encoding: [0xdf,0x85,A,A]
                      # CHECK:                                 #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
                      # CHECK: daddiu $5, $5, 8                # encoding: [0x64,0xa5,0x00,0x08]
                      # CHECK: daddu $5, $5, $6                # encoding: [0x00,0xa6,0x28,0x2d]
dla $6, symbol+8($6)  # CHECK: ld $1, %got_disp(symbol)($gp)   # encoding: [0xdf,0x81,A,A]
                      # CHECK:                                 #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
                      # CHECK: daddiu $1, $1, 8                # encoding: [0x64,0x21,0x00,0x08]
                      # CHECK: daddu $6, $1, $6                # encoding: [0x00,0x26,0x30,0x2d]
dla $5, 1f            # CHECK: ld $5, %got_disp(.Ltmp0)($gp)   # encoding: [0xdf,0x85,A,A]
                      # CHECK:                                 #   fixup A - offset: 0, value: %got_disp(.Ltmp0), kind: fixup_Mips_GOT_DISP
1:

# PIC expansions involving $25 are special.
dla $25, symbol       # CHECK: ld $25, %call16(symbol)($gp)     # encoding: [0xdf,0x99,A,A]
                      # CHECK:                                  #   fixup A - offset: 0, value: %call16(symbol), kind: fixup_Mips_CALL16
dla $25, symbol($6)   # CHECK: ld $25, %got_disp(symbol)($gp)   # encoding: [0xdf,0x99,A,A]
                      # CHECK:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
                      # CHECK: daddu $25, $25, $6               # encoding: [0x03,0x26,0xc8,0x2d]
dla $25, symbol($25)  # CHECK: ld $1, %got_disp(symbol)($gp)    # encoding: [0xdf,0x81,A,A]
                      # CHECK:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
                      # CHECK: daddu $25, $1, $25               # encoding: [0x00,0x39,0xc8,0x2d]
dla $25, symbol+8     # CHECK: ld $25, %got_disp(symbol)($gp)   # encoding: [0xdf,0x99,A,A]
                      # CHECK:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
                      # CHECK: daddiu $25, $25, 8               # encoding: [0x67,0x39,0x00,0x08]
dla $25, symbol+8($6) # CHECK: ld $25, %got_disp(symbol)($gp)   # encoding: [0xdf,0x99,A,A]
                      # CHECK:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
                      # CHECK: daddiu $25, $25, 8               # encoding: [0x67,0x39,0x00,0x08]
                      # CHECK: daddu $25, $25, $6               # encoding: [0x03,0x26,0xc8,0x2d]
dla $25, symbol+8($25)# CHECK: ld $1, %got_disp(symbol)($gp)    # encoding: [0xdf,0x81,A,A]
                      # CHECK:                                  #   fixup A - offset: 0, value: %got_disp(symbol), kind: fixup_Mips_GOT_DISP
                      # CHECK: daddiu $1, $1, 8                 # encoding: [0x64,0x21,0x00,0x08]
                      # CHECK: daddu $25, $1, $25               # encoding: [0x00,0x39,0xc8,0x2d]
dla $25, 1f           # CHECK: ld $25, %got_disp(.Ltmp1)($gp)    # encoding: [0xdf,0x99,A,A]
                      # CHECK:                                  #   fixup A - offset: 0, value: %got_disp(.Ltmp1), kind: fixup_Mips_GOT_DISP
1:
