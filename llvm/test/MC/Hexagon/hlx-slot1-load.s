/// Tests that an HLX packet can have a single load in slot 1.

// RUN: llvm-mc -triple=hexagon -mv85 -mattr=+hlx %s -filetype=obj | \
// RUN:   llvm-objdump -d --mcpu=hexagonv85 --mattr=+hlx - | FileCheck %s

{
  l7.tmp = mem(r12)
  l0 = not(l7)
}

//      CHECK: e50075e0 {      L0 = not(L7)
// CHECK-NEXT: 980cd007        L7.tmp = mem(r12) }
