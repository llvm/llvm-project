// RUN: not llvm-mc -triple=hexagon -mv85 -mattr=+hlx -filetype=asm %s 2>/dev/null | FileCheck %s
// RUN: not llvm-mc -triple=hexagon -mv85 -mattr=+hlx -filetype=asm %s 2>&1 >/dev/null | \
// RUN:   FileCheck %s --check-prefix=ERR

/// HLX loads use slot 1, HLX multiplies use slot 3, and other HLX compute
/// instructions use slot 2.

{ L0.h = L0.h
  r0 = add(r19,#2048)
  r19 = add(r16,#2048)
  L0.tmp = mem(r19) }
// CHECK:      {
// CHECK-NEXT:   L0.h = L0.h
// CHECK-NEXT:   r0 = add(r19,#2048)
// CHECK-NEXT:   L0.tmp = mem(r19)
// CHECK-NEXT:   r19 = add(r16,#2048)
// CHECK-NEXT: }

// ERR: :[[@LINE+2]]:{{[0-9]+}}: error: invalid instruction packet: slot error
{ l7.tmp = mem(r12)
  l1.tmp = mem(r3) }

// ERR: :[[@LINE+2]]:{{[0-9]+}}: error: invalid instruction packet: slot error
{ if (p0) l7.tmp = mem(r12)
  l1.tmp = mem(r3) }

// ERR: :[[@LINE+2]]:{{[0-9]+}}: error: invalid instruction packet: slot error
{ L7:6.w = mpy(L4.h,r3.uh)
  L15:14.w = mpy(L4.h,r4.h) }

// ERR: :[[@LINE+2]]:{{[0-9]+}}: error: invalid instruction packet: slot error
{ L6.uh = absdiff(L5.uh,L3.uh)
  L9:8.h = add(L1:0.h,L3:2.h) }

// ERR: :[[@LINE+2]]:{{[0-9]+}}: error: invalid instruction packet: slot error
{ L1.h += shr(L2.h,L3.h)
  L4.h += shr(L5.h,L6.h) }

// ERR: :[[@LINE+2]]:{{[0-9]+}}: error: invalid instruction packet: slot error
{ L1:0.w += mpy(L2.h,L3.h)
  L5:4.w += mpy(L6.h,L7.h) }
