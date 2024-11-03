// RUN: llvm-mc -triple=amdgcn -mcpu=tonga -show-encoding %s | FileCheck -check-prefix=VI %s

buffer_wbinvl1_vol
// VI: buffer_wbinvl1_vol ; encoding: [0x00,0x00,0xfc,0xe0,0x00,0x00,0x00,0x00]
