// RUN: not llvm-mc -triple=hexagon --mcpu=hexagonv85 --mattr=+hlx -filetype=asm %s 2>&1 | \
// RUN:   FileCheck %s --implicit-check-not=error

// CHECK: error: unrecognized instruction
// CHECK-NEXT: l31.uh=max(l1.uh,l2.uh)
{
  l31.uh=max(l1.uh,l2.uh)
}

// CHECK: error: invalid operand for instruction
// CHECK-NEXT: l0.uh=add(l16.uh,l15.uh)
{
  l0.uh=add(l16.uh,l15.uh)
}

{
  l4.uh=max(l5.uh,l6.uh)
}
