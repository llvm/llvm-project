# RUN: llvm-mc %s -triple=mips64 -mcpu=mips64r2 -mattr=+msa -show-encoding | FileCheck %s
#
# CHECK:        insert.d        $w1[1], $sp            # encoding: [0x79,0x39,0xe8,0x59]

                insert.d        $w1[1], $sp
