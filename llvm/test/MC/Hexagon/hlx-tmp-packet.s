// RUN: llvm-mc %s -triple=hexagon -mv85 -mattr=+hlx -filetype=asm | FileCheck %s

// Check that HLX packets like the one below do not produce
// "register modified more than once" errors for .tmp registers.

{
  l4.tmp = mem(r0)
  l5:4.w+=shr(l3:2.w, l1.h)
}

//      CHECK: {
// CHECK-NEXT:   L5:4.w += shr(L3:2.w,L1.h)
// CHECK-NEXT:   L4.tmp = mem(r0)
// CHECK-NEXT: }
