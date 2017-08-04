@ RUN: not llvm-mc -triple=thumbv7m--none-eabi < %s 2>&1 | FileCheck %s

@ These instructions all write to the PC, so are UNPREDICTABLE if they are in
@ an IT block, but not the last instruction in the block.

  itttt eq
  addeq pc, r0
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  addeq pc, sp, pc
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  beq.n #.+0x20
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  nopeq
  itttt eq
  beq.w #.+0x20
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  bleq sym
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  blxeq r0
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  nopeq
  itttt eq
  bxeq r0
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  ldmeq r0, {r8, pc}
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  ldmdbeq r0, {r8, pc}
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  nopeq
  itttt eq
  ldreq pc, [r0, #4]
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  ldreq pc, [r0, #-4]
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  ldreq pc, [pc, #4]
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  nopeq
  itttt eq
  ldreq pc, [r0, r1, LSL #1]
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  moveq pc, r0
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  popeq {r0, pc}
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  nopeq
  itttt eq
  popeq {r8, pc}
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  popeq {pc}
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  tbbeq [r0, r1]
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  nopeq
  itt eq
  tbheq [r0, r1, LSL #1]
@ CHECK: :[[@LINE-1]]:{{[0-9]+}}: error: instruction must be outside of IT block or the last instruction in an IT block
  nopeq
