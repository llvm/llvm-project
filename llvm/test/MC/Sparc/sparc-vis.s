! RUN: llvm-mc %s -triple=sparcv9 -mcpu=niagara -show-encoding | FileCheck %s

        ! CHECK: fzeros %f31   ! encoding: [0xbf,0xb0,0x0c,0x20]
        fzeros %f31
