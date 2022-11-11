; Check that the GHC calling convention works (s390x)
;
; RUN: llc -mtriple=s390x-ibm-linux < %s | FileCheck %s

@base  = external dso_local global i64 ; assigned to register: r7
@sp    = external dso_local global i64 ; assigned to register: r8
@hp    = external dso_local global i64 ; assigned to register: r10
@r1    = external dso_local global i64 ; assigned to register: r11
@r2    = external dso_local global i64 ; assigned to register: r12
@r3    = external dso_local global i64 ; assigned to register: r13
@r4    = external dso_local global i64 ; assigned to register: r6
@r5    = external dso_local global i64 ; assigned to register: r2
@r6    = external dso_local global i64 ; assigned to register: r3
@r7    = external dso_local global i64 ; assigned to register: r4
@r8    = external dso_local global i64 ; assigned to register: r5
@splim = external dso_local global i64 ; assigned to register: r9

@f1 = external dso_local global float  ; assigned to register: s8
@f2 = external dso_local global float  ; assigned to register: s9
@f3 = external dso_local global float  ; assigned to register: s10
@f4 = external dso_local global float  ; assigned to register: s11
@f5 = external dso_local global float  ; assigned to register: s0
@f6 = external dso_local global float  ; assigned to register: s1

@d1 = external dso_local global double ; assigned to register: d12
@d2 = external dso_local global double ; assigned to register: d13
@d3 = external dso_local global double ; assigned to register: d14
@d4 = external dso_local global double ; assigned to register: d15
@d5 = external dso_local global double ; assigned to register: d2
@d6 = external dso_local global double ; assigned to register: d3

define ghccc void @foo() nounwind {
entry:
  ; CHECK:      larl    {{%r[0-9]+}}, d6
  ; CHECK-NEXT: ld      %f3, 0({{%r[0-9]+}})
  ; CHECK-NEXT: larl    {{%r[0-9]+}}, d5
  ; CHECK-NEXT: ld      %f2, 0({{%r[0-9]+}})
  ; CHECK-NEXT: larl    {{%r[0-9]+}}, d4
  ; CHECK-NEXT: ld      %f15, 0({{%r[0-9]+}})
  ; CHECK-NEXT: larl    {{%r[0-9]+}}, d3
  ; CHECK-NEXT: ld      %f14, 0({{%r[0-9]+}})
  ; CHECK-NEXT: larl    {{%r[0-9]+}}, d2
  ; CHECK-NEXT: ld      %f13, 0({{%r[0-9]+}})
  ; CHECK-NEXT: larl    {{%r[0-9]+}}, d1
  ; CHECK-NEXT: ld      %f12, 0({{%r[0-9]+}})
  ; CHECK-NEXT: larl    {{%r[0-9]+}}, f6
  ; CHECK-NEXT: le      %f1, 0({{%r[0-9]+}})
  ; CHECK-NEXT: larl    {{%r[0-9]+}}, f5
  ; CHECK-NEXT: le      %f0, 0({{%r[0-9]+}})
  ; CHECK-NEXT: larl    {{%r[0-9]+}}, f4
  ; CHECK-NEXT: le      %f11, 0({{%r[0-9]+}})
  ; CHECK-NEXT: larl    {{%r[0-9]+}}, f3
  ; CHECK-NEXT: le      %f10, 0({{%r[0-9]+}})
  ; CHECK-NEXT: larl    {{%r[0-9]+}}, f2
  ; CHECK-NEXT: le      %f9, 0({{%r[0-9]+}})
  ; CHECK-NEXT: larl    {{%r[0-9]+}}, f1
  ; CHECK-NEXT: le      %f8, 0({{%r[0-9]+}})
  ; CHECK-NEXT: lgrl    %r9,  splim
  ; CHECK-NEXT: lgrl    %r5,  r8
  ; CHECK-NEXT: lgrl    %r4,  r7
  ; CHECK-NEXT: lgrl    %r3,  r6
  ; CHECK-NEXT: lgrl    %r2,  r5
  ; CHECK-NEXT: lgrl    %r6,  r4
  ; CHECK-NEXT: lgrl    %r13, r3
  ; CHECK-NEXT: lgrl    %r12, r2
  ; CHECK-NEXT: lgrl    %r11, r1
  ; CHECK-NEXT: lgrl    %r10, hp
  ; CHECK-NEXT: lgrl    %r8,  sp
  ; CHECK-NEXT: lgrl    %r7,  base
  %0  = load double, ptr @d6
  %1  = load double, ptr @d5
  %2  = load double, ptr @d4
  %3  = load double, ptr @d3
  %4  = load double, ptr @d2
  %5  = load double, ptr @d1
  %6  = load float, ptr @f6
  %7  = load float, ptr @f5
  %8  = load float, ptr @f4
  %9  = load float, ptr @f3
  %10 = load float, ptr @f2
  %11 = load float, ptr @f1
  %12 = load i64, ptr @splim
  %13 = load i64, ptr @r8
  %14 = load i64, ptr @r7
  %15 = load i64, ptr @r6
  %16 = load i64, ptr @r5
  %17 = load i64, ptr @r4
  %18 = load i64, ptr @r3
  %19 = load i64, ptr @r2
  %20 = load i64, ptr @r1
  %21 = load i64, ptr @hp
  %22 = load i64, ptr @sp
  %23 = load i64, ptr @base
  ; CHECK: brasl %r14, bar
  tail call ghccc void @bar(i64 %23, i64 %22, i64 %21, i64 %20, i64 %19, i64 %18, i64 %17, i64 %16, i64 %15, i64 %14, i64 %13, i64 %12,
                            float %11, float %10, float %9, float %8, float %7, float %6,
                            double %5, double %4, double %3, double %2, double %1, double %0) nounwind
  ret void
}

declare ghccc void @bar(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64,
                        float, float, float, float, float, float,
                        double, double, double, double, double, double)
