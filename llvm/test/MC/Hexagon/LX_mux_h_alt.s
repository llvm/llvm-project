// RUN: llvm-mc %s -filetype=asm -triple=hexagon --mcpu=hexagonv85 \
// RUN:   --mattr=+hlx | FileCheck %s

// Check the mapping to LX_mux_h and the order of its operands.

if (L1.h<0) L2.h = L3.h

// CHECK: L2.h = mux(L2.h,L3.h,L1.h)
