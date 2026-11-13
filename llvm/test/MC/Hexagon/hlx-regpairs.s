// RUN: not llvm-mc -triple=hexagon --mcpu=hexagonv85 --mattr=+hlx -filetype=asm %s 2>&1 | \
// RUN:   FileCheck %s --implicit-check-not=error

// CHECK: error: unexpected token
// CHECK-NEXT: l25:24.h=max(l1:0.h,l3:2.h)
{
  l25:24.h=max(l1:0.h,l3:2.h)
}

// CHECK: error: invalid operand for instruction
// CHECK-NEXT: l9:8.h=add(l17:16.h,l5:4.h)
{
  l9:8.h=add(l17:16.h,l5:4.h)
}

{
  l11:10.h=max(l13:12.h,l15:14.h)
}

// CHECK: error: unexpected token
// CHECK-NEXT: l24:25.h=max(l0:1.h,l2:3.h)
{
  l24:25.h=max(l0:1.h,l2:3.h)
}

// CHECK: error: unexpected token
// CHECK-NEXT: l8:9.h=add(l16:17.h,l4:5.h)
{
  l8:9.h=add(l16:17.h,l4:5.h)
}

// CHECK: error: unexpected token
// CHECK-NEXT: l10:11.h=max(l12:13.h,l14:15.h)
{
  l10:11.h=max(l12:13.h,l14:15.h)
}

// CHECK: error: unexpected token
// CHECK-NEXT: l0:1.h=max(l1:0.h,l3:2.h)
{
  l0:1.h=max(l1:0.h,l3:2.h)
}
