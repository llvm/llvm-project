// RUN: llvm-mc -triple=hexagon --mcpu=hexagonv85 --mattr=+hlx -filetype=obj %s | \
// RUN:   llvm-objdump --mcpu=hexagonv85 --mattr=+hlx -d - | FileCheck %s --implicit-check-not=error

l1:0.h = max(l1:0.h,l3:2.h)
// CHECK: ef04f820 { L1:0.h = max(L1:0.h,L3:2.h) }

l7:6.w = add(mpy(l5.uh,hi(l7:6).h),l9:8.w)
// CHECK-NEXT: e7f0f56d { L7:6.w = add(mpy(L5.uh,hi(L7:6).h),L9:8.w) }

l11:10.w = lut4(l5.uh,r13:12.uh):>>1
// CHECK-NEXT: ed0ce155 { L11:10.w = lut4(L5.uh,r13:12.uh):>>1 }

l9:8.w = lut4(l5.uh,r9:8.uh):>>1:neg
// CHECK-NEXT: ed08e170 { L9:8.w = lut4(L5.uh,r9:8.uh):>>1:neg }
