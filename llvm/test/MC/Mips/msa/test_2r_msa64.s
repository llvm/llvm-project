# RUN: llvm-mc %s -triple=mips64 -mcpu=mips64r2 -mattr=+msa -show-encoding | FileCheck %s
#
# CHECK:        fill.d  $w27, $9                # encoding: [0x7b,0x03,0x4e,0xde]

                fill.d  $w27, $9
