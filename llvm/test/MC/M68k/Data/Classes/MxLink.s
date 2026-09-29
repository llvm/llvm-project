; RUN: llvm-mc --show-encoding -triple=m68k %s | FileCheck %s

# CHECK:      link.w %a2, #1023
# CHECK-SAME: encoding: [0x4e,0x52,0x03,0xff]
link.w %a2, #1023

# CHECK:      link.l %a1, #1073741823
# CHECK-SAME: encoding: [0x48,0x09,0x3f,0xff,0xff,0xff]
link.l %a1, #1073741823

# CHECK:      unlk %a5
# CHECK-SAME: encoding: [0x4e,0x5d]
unlk %a5