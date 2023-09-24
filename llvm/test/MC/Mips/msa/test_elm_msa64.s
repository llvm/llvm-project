# RUN: llvm-mc %s -triple=mips64 -mcpu=mips64r2 -mattr=+msa -show-encoding | FileCheck %s

copy_s.d $19, $w31[0] # CHECK: copy_s.d $19, $w31[0] # encoding: [0x78,0xb8,0xfc,0xd9]
