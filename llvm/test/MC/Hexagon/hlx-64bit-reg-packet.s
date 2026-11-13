// An HLX instruction that reads a 64-bit scalar register may not be in the
// same packet with another HLX instruction that uses a scalar register or
// immediate. HLX loads and stores, and non-HLX instructions, are not
// restricted.

// RUN: not llvm-mc -triple=hexagon -mv85 -mattr=+hlx -filetype=asm %s 2>&1 | \
// RUN:   FileCheck %s --implicit-check-not=error --implicit-check-not=note

// 64-bit scalar register with a 32-bit scalar register.
{
  l0.h = lut4(l1.uh,r1:0.h)
  l2.h = shr(l3.h,r4.h)
}
// CHECK: :[[@LINE-3]]:3: error: HLX instruction using a 64-bit scalar register cannot be in the same packet with another HLX instruction using a scalar register or immediate

// HLX instruction reading a 32-bit scalar register.
{
  l0.h = r5.h
  l2.h = lut4(l3.uh,r1:0.h)
}
// CHECK: :[[@LINE-2]]:3: error: HLX instruction using a 64-bit scalar register cannot be in the same packet with another HLX instruction using a scalar register or immediate

// 64-bit scalar register with a 32-bit scalar immediate.
{
  l0.h = lut4(l1.uh,r1:0.h)
  L2.h = #5
}
// CHECK: :[[@LINE-3]]:3: error: HLX instruction using a 64-bit scalar register cannot be in the same packet with another HLX instruction using a scalar register or immediate

// Two instructions that use 64-bit scalar registers.
{
  l0.h = lut4(l1.uh,r1:0.h)
  l2.h = lut4(l3.uh,r3:2.h)
}
// CHECK: :[[@LINE-3]]:3: error: HLX instruction using a 64-bit scalar register cannot be in the same packet with another HLX instruction using a scalar register or immediate

// Valid: HLX loads and stores are excluded.
{
  l0.h = lut4(l1.uh,r1:0.h)
  l2.tmp = mem(r2)
  l2.h = l2.h
}
{
  l0.h = lut4(l1.uh,r1:0.h)
  l3.h = l3.h
  mem(r2) = l3.new
}

// Valid: the other HLX instruction does not use a scalar register.
{
  l0.h = lut4(l1.uh,r1:0.h)
  l2.h = l3.h
}

// Valid: non-HLX instruction using scalar registers.
{
  l0.h = lut4(l1.uh,r1:0.h)
  r3:2 = add(r5:4,r7:6)
}

// Valid: two HLX instructions using 32-bit scalar registers.
{
  l0.h = r5.h
  l2.h = shr(l3.h,r4.h)
}
