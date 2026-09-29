# RUN: llvm-mc -show-encoding -mcpu=mips32 -triple mips-unknown-unknown %s | FileCheck -check-prefix=BE %s
# RUN: llvm-mc -show-encoding -mcpu=mips32 -triple mipsel-unknown-unknown %s | FileCheck -check-prefix=LE %s
#
        .text
        b foo # BE: b foo # encoding: [0x10,0x00,A,A]
              # LE: b foo # encoding: [A,A,0x00,0x10]
