// Basic new-value checks for HLX packets.

// RUN: not llvm-mc -mv85 -mattr=+hlx -triple=hexagon -filetype=asm %s 2>&1 | \
// RUN:   FileCheck %s --implicit-check-not=error

{
  nop
  mem(r0)=l1.new
}

// CHECK: error: New value register consumer has no producer
