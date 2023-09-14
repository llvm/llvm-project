# RUN: llvm-mc %s -triple=mips -mcpu=mips32r2 -mattr=+msa -show-encoding | FileCheck %s
#
#CHECK:      bnz.b        $w0, 4        # encoding: [0x47,0x80,0x00,0x01]
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]
#CHECK:      bnz.h        $w1, 16       # encoding: [0x47,0xa1,0x00,0x04]
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]
#CHECK:      bnz.w        $w2, 128      # encoding: [0x47,0xc2,0x00,0x20]
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]
#CHECK:      bnz.d        $w3, -128     # encoding: [0x47,0xe3,0xff,0xe0]
#CHECK:      bnz.b        $w0, SYMBOL0  # encoding: [0x47,0x80,A,A]
                                        #   fixup A - offset: 0, value: SYMBOL0, kind: fixup_Mips_PC16
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]
#CHECK:      bnz.h        $w1, SYMBOL1  # encoding: [0x47,0xa1,A,A]
                                        #   fixup A - offset: 0, value: SYMBOL1, kind: fixup_Mips_PC16
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]
#CHECK:      bnz.w        $w2, SYMBOL2  # encoding: [0x47,0xc2,A,A]
                                        #   fixup A - offset: 0, value: SYMBOL2, kind: fixup_Mips_PC16
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]
#CHECK:      bnz.d        $w3, SYMBOL3  # encoding: [0x47,0xe3,A,A]
                                        #   fixup A - offset: 0, value: SYMBOL3, kind: fixup_Mips_PC16
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]

#CHECK:      bnz.v        $w0, 4        # encoding: [0x45,0xe0,0x00,0x01]
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]
#CHECK:      bnz.v        $w0, SYMBOL0  # encoding: [0x45,0xe0,A,A]
                                        #   fixup A - offset: 0, value: SYMBOL0, kind: fixup_Mips_PC16
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]

#CHECK:      bz.b         $w0, 128      # encoding: [0x47,0x00,0x00,0x20]
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]
#CHECK:      bz.h         $w1, 256      # encoding: [0x47,0x21,0x00,0x40]
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]
#CHECK:      bz.w         $w2, 512      # encoding: [0x47,0x42,0x00,0x80]
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]
#CHECK:      bz.d         $w3, -1024    # encoding: [0x47,0x63,0xff,0x00]
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]
#CHECK:      bz.b         $w0, SYMBOL0  # encoding: [0x47,0x00,A,A]
                                        #   fixup A - offset: 0, value: SYMBOL0, kind: fixup_Mips_PC16
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]
#CHECK:      bz.h         $w1, SYMBOL1  # encoding: [0x47,0x21,A,A]
                                        #   fixup A - offset: 0, value: SYMBOL1, kind: fixup_Mips_PC16
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]
#CHECK:      bz.w         $w2, SYMBOL2  # encoding: [0x47,0x42,A,A]
                                        #   fixup A - offset: 0, value: SYMBOL2, kind: fixup_Mips_PC16
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]
#CHECK:      bz.d         $w3, SYMBOL3  # encoding: [0x47,0x63,A,A]
                                        #   fixup A - offset: 0, value: SYMBOL3, kind: fixup_Mips_PC16
#CHECK:      nop                        # encoding: [0x00,0x00,0x00,0x00]

#CHECK:      bz.v        $w0, 4        # encoding: [0x45,0x60,0x00,0x01]
#CHECK:      nop                       # encoding: [0x00,0x00,0x00,0x00]
#CHECK:      bz.v        $w0, SYMBOL0  # encoding: [0x45,0x60,A,A]
                                       #   fixup A - offset: 0, value: SYMBOL0, kind: fixup_Mips_PC16
#CHECK:      nop                       # encoding: [0x00,0x00,0x00,0x00]

bnz.b        $w0, 4
bnz.h        $w1, 16
bnz.w        $w2, 128
bnz.d        $w3, -128
bnz.b        $w0, SYMBOL0
bnz.h        $w1, SYMBOL1
bnz.w        $w2, SYMBOL2
bnz.d        $w3, SYMBOL3

bnz.v        $w0, 4
bnz.v        $w0, SYMBOL0

bz.b        $w0, 128
bz.h        $w1, 256
bz.w        $w2, 512
bz.d        $w3, -1024
bz.b        $w0, SYMBOL0
bz.h        $w1, SYMBOL1
bz.w        $w2, SYMBOL2
bz.d        $w3, SYMBOL3

bz.v        $w0, 4
bz.v        $w0, SYMBOL0
