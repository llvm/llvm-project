; RUN: llc < %s -march=mipsel -relocation-model=pic | FileCheck %s
; RUN: llc < %s -mtriple=mipsel-none-nacl-gnu -relocation-model=pic -mips-tail-calls=1\
; RUN:  | FileCheck %s -check-prefix=CHECK-NACL
; RUN: llc < %s -march=mipsel -mcpu=mips32 -mattr=+nooddspreg -relocation-model=pic -mips-tail-calls=1| FileCheck %s -check-prefix=NOODDSPREG
; RUN: llc < %s -march=mipsel -mcpu=mips32r2 -mattr=+fp64,+nooddspreg -relocation-model=pic -mips-tail-calls=1 | FileCheck %s -check-prefix=FP64-NOODDSPREG


@gi0 = external global i32
@gi1 = external global i32
@gi2 = external global i32
@gi3 = external global i32
@gi4 = external global i32
@gi5 = external global i32
@gi6 = external global i32
@gi7 = external global i32
@gi8 = external global i32
@gi9 = external global i32
@gi10 = external global i32
@gi11 = external global i32
@gi12 = external global i32
@gi13 = external global i32
@gi14 = external global i32
@gi15 = external global i32
@gi16 = external global i32
@gfa0 = external global float
@gfa1 = external global float
@gfa2 = external global float
@gfa3 = external global float
@gfa4 = external global float
@gfa5 = external global float
@gfa6 = external global float
@gfa7 = external global float
@gfa8 = external global float
@gfa9 = external global float
@gfa10 = external global float
@gfa11 = external global float
@gfa12 = external global float
@gfa13 = external global float
@gfa14 = external global float
@gfa15 = external global float
@gfa16 = external global float
@gfa17 = external global float
@gfa18 = external global float
@gfa19 = external global float
@gfa20 = external global float
@gf0 = external global float
@gf1 = external global float
@gf2 = external global float
@gf3 = external global float
@gf4 = external global float
@gf5 = external global float
@gf6 = external global float
@gf7 = external global float
@gf8 = external global float
@gf9 = external global float
@gf10 = external global float
@gf11 = external global float
@gf12 = external global float
@gf13 = external global float
@gf14 = external global float
@gf15 = external global float
@gf16 = external global float
@gf17 = external global float
@gf18 = external global float
@gf19 = external global float
@gf20 = external global float
@g0 = external global i32
@g1 = external global i32
@g2 = external global i32
@g3 = external global i32
@g4 = external global i32
@g5 = external global i32
@g6 = external global i32
@g7 = external global i32
@g8 = external global i32
@g9 = external global i32
@g10 = external global i32
@g11 = external global i32
@g12 = external global i32
@g13 = external global i32
@g14 = external global i32
@g15 = external global i32
@g16 = external global i32

@fa = common global [11 x float] zeroinitializer, align 4
@da = common global [11 x double] zeroinitializer, align 8

define void @caller0() nounwind {
entry:
; CHECK: caller0
; CHECK: lw  $3
; CHECK: lw  $24
; CHECK: lw  $15
; CHECK: lw  $14
; CHECK: lw  $13
; CHECK: lw  $12
; CHECK: lw  $11
; CHECK: lw  $10
; CHECK: lw  $9
; CHECK: lw  $8
; CHECK: lw  $7
; CHECK: lw  $6
; CHECK: lw  $5
; CHECK: lw  $4

; t6, t7 and t8 are reserved in NaCl and cannot be used for fastcc.
; CHECK-NACL-NOT: lw  $14
; CHECK-NACL-NOT: lw  $15
; CHECK-NACL-NOT: lw  $24

  %0 = load i32, ptr @gi0, align 4
  %1 = load i32, ptr @gi1, align 4
  %2 = load i32, ptr @gi2, align 4
  %3 = load i32, ptr @gi3, align 4
  %4 = load i32, ptr @gi4, align 4
  %5 = load i32, ptr @gi5, align 4
  %6 = load i32, ptr @gi6, align 4
  %7 = load i32, ptr @gi7, align 4
  %8 = load i32, ptr @gi8, align 4
  %9 = load i32, ptr @gi9, align 4
  %10 = load i32, ptr @gi10, align 4
  %11 = load i32, ptr @gi11, align 4
  %12 = load i32, ptr @gi12, align 4
  %13 = load i32, ptr @gi13, align 4
  %14 = load i32, ptr @gi14, align 4
  %15 = load i32, ptr @gi15, align 4
  %16 = load i32, ptr @gi16, align 4
  tail call fastcc void @callee0(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10, i32 %11, i32 %12, i32 %13, i32 %14, i32 %15, i32 %16)
  ret void
}

define internal fastcc void @callee0(i32 %a0, i32 %a1, i32 %a2, i32 %a3, i32 %a4, i32 %a5, i32 %a6, i32 %a7, i32 %a8, i32 %a9, i32 %a10, i32 %a11, i32 %a12, i32 %a13, i32 %a14, i32 %a15, i32 %a16) nounwind noinline {
entry:
; CHECK: callee0
; CHECK-DAG: sw  $4
; CHECK-DAG: sw  $5
; CHECK-DAG: sw  $7
; CHECK-DAG: sw  $8
; CHECK-DAG: sw  $9
; CHECK-DAG: sw  $10
; CHECK-DAG: sw  $11
; CHECK-DAG: sw  $12
; CHECK-DAG: sw  $13
; CHECK-DAG: sw  $14
; CHECK-DAG: sw  $15
; CHECK-DAG: sw  $24
; CHECK-DAG: sw  $3

; t6, t7 and t8 are reserved in NaCl and cannot be used for fastcc.
; CHECK-NACL-NOT: sw  $14
; CHECK-NACL-NOT: sw  $15
; CHECK-NACL-NOT: sw  $24

  store i32 %a0, ptr @g0, align 4
  store i32 %a1, ptr @g1, align 4
  store i32 %a2, ptr @g2, align 4
  store i32 %a3, ptr @g3, align 4
  store i32 %a4, ptr @g4, align 4
  store i32 %a5, ptr @g5, align 4
  store i32 %a6, ptr @g6, align 4
  store i32 %a7, ptr @g7, align 4
  store i32 %a8, ptr @g8, align 4
  store i32 %a9, ptr @g9, align 4
  store i32 %a10, ptr @g10, align 4
  store i32 %a11, ptr @g11, align 4
  store i32 %a12, ptr @g12, align 4
  store i32 %a13, ptr @g13, align 4
  store i32 %a14, ptr @g14, align 4
  store i32 %a15, ptr @g15, align 4
  store i32 %a16, ptr @g16, align 4
  ret void
}

define void @caller1(float %a0, float %a1, float %a2, float %a3, float %a4, float %a5, float %a6, float %a7, float %a8, float %a9, float %a10, float %a11, float %a12, float %a13, float %a14, float %a15, float %a16, float %a17, float %a18, float %a19, float %a20) nounwind {
entry:
; CHECK: caller1
; CHECK: lwc1  $f19
; CHECK: lwc1  $f18
; CHECK: lwc1  $f17
; CHECK: lwc1  $f16
; CHECK: lwc1  $f15
; CHECK: lwc1  $f14
; CHECK: lwc1  $f13
; CHECK: lwc1  $f12
; CHECK: lwc1  $f11
; CHECK: lwc1  $f10
; CHECK: lwc1  $f9
; CHECK: lwc1  $f8
; CHECK: lwc1  $f7
; CHECK: lwc1  $f6
; CHECK: lwc1  $f5
; CHECK: lwc1  $f4
; CHECK: lwc1  $f3
; CHECK: lwc1  $f2
; CHECK: lwc1  $f1
; CHECK: lwc1  $f0

  %0 = load float, ptr @gfa0, align 4
  %1 = load float, ptr @gfa1, align 4
  %2 = load float, ptr @gfa2, align 4
  %3 = load float, ptr @gfa3, align 4
  %4 = load float, ptr @gfa4, align 4
  %5 = load float, ptr @gfa5, align 4
  %6 = load float, ptr @gfa6, align 4
  %7 = load float, ptr @gfa7, align 4
  %8 = load float, ptr @gfa8, align 4
  %9 = load float, ptr @gfa9, align 4
  %10 = load float, ptr @gfa10, align 4
  %11 = load float, ptr @gfa11, align 4
  %12 = load float, ptr @gfa12, align 4
  %13 = load float, ptr @gfa13, align 4
  %14 = load float, ptr @gfa14, align 4
  %15 = load float, ptr @gfa15, align 4
  %16 = load float, ptr @gfa16, align 4
  %17 = load float, ptr @gfa17, align 4
  %18 = load float, ptr @gfa18, align 4
  %19 = load float, ptr @gfa19, align 4
  %20 = load float, ptr @gfa20, align 4
  tail call fastcc void @callee1(float %0, float %1, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9, float %10, float %11, float %12, float %13, float %14, float %15, float %16, float %17, float %18, float %19, float %20)
  ret void
}

define internal fastcc void @callee1(float %a0, float %a1, float %a2, float %a3, float %a4, float %a5, float %a6, float %a7, float %a8, float %a9, float %a10, float %a11, float %a12, float %a13, float %a14, float %a15, float %a16, float %a17, float %a18, float %a19, float %a20) nounwind noinline {
entry:
; CHECK-LABEL: callee1:
; CHECK-DAG: swc1  $f17
; CHECK-DAG: swc1  $f16
; CHECK-DAG: swc1  $f15
; CHECK-DAG: swc1  $f14
; CHECK-DAG: swc1  $f13
; CHECK-DAG: swc1  $f12
; CHECK-DAG: swc1  $f11
; CHECK-DAG: swc1  $f10
; CHECK-DAG: swc1  $f9
; CHECK-DAG: swc1  $f8
; CHECK-DAG: swc1  $f7
; CHECK-DAG: swc1  $f6
; CHECK-DAG: swc1  $f5
; CHECK-DAG: swc1  $f4
; CHECK-DAG: swc1  $f3
; CHECK-DAG: swc1  $f2
; CHECK-DAG: swc1  $f1
; CHECK-DAG: swc1  $f0
; CHECK-DAG: swc1  $f18
; CHECK-DAG: swc1  $f19

  store float %a0, ptr @gf0, align 4
  store float %a1, ptr @gf1, align 4
  store float %a2, ptr @gf2, align 4
  store float %a3, ptr @gf3, align 4
  store float %a4, ptr @gf4, align 4
  store float %a5, ptr @gf5, align 4
  store float %a6, ptr @gf6, align 4
  store float %a7, ptr @gf7, align 4
  store float %a8, ptr @gf8, align 4
  store float %a9, ptr @gf9, align 4
  store float %a10, ptr @gf10, align 4
  store float %a11, ptr @gf11, align 4
  store float %a12, ptr @gf12, align 4
  store float %a13, ptr @gf13, align 4
  store float %a14, ptr @gf14, align 4
  store float %a15, ptr @gf15, align 4
  store float %a16, ptr @gf16, align 4
  store float %a17, ptr @gf17, align 4
  store float %a18, ptr @gf18, align 4
  store float %a19, ptr @gf19, align 4
  store float %a20, ptr @gf20, align 4
  ret void
}

define void @caller2() {
entry:

; NOODDSPREG-LABEL:  caller2:

; Check that first 10 arguments are passed in even float registers
; f0, f2, ... , f18. Check that 11th argument is passed on stack.

; NOODDSPREG-DAG:    lw      $[[R0:[0-9]+]], %got(fa)(${{[0-9]+|gp}})
; NOODDSPREG-DAG:    lwc1    $f0, 0($[[R0]])
; NOODDSPREG-DAG:    lwc1    $f2, 4($[[R0]])
; NOODDSPREG-DAG:    lwc1    $f4, 8($[[R0]])
; NOODDSPREG-DAG:    lwc1    $f6, 12($[[R0]])
; NOODDSPREG-DAG:    lwc1    $f8, 16($[[R0]])
; NOODDSPREG-DAG:    lwc1    $f10, 20($[[R0]])
; NOODDSPREG-DAG:    lwc1    $f12, 24($[[R0]])
; NOODDSPREG-DAG:    lwc1    $f14, 28($[[R0]])
; NOODDSPREG-DAG:    lwc1    $f16, 32($[[R0]])
; NOODDSPREG-DAG:    lwc1    $f18, 36($[[R0]])

; NOODDSPREG-DAG:    lwc1    $[[F0:f[0-9]*[02468]]], 40($[[R0]])
; NOODDSPREG-DAG:    swc1    $[[F0]], 0($sp)

  %0 = load float, ptr @fa, align 4
  %1 = load float, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 1), align 4
  %2 = load float, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 2), align 4
  %3 = load float, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 3), align 4
  %4 = load float, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 4), align 4
  %5 = load float, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 5), align 4
  %6 = load float, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 6), align 4
  %7 = load float, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 7), align 4
  %8 = load float, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 8), align 4
  %9 = load float, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 9), align 4
  %10 = load float, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 10), align 4
  tail call fastcc void @callee2(float %0, float %1, float %2, float %3,
                                 float %4, float %5, float %6, float %7,
                                 float %8, float %9, float %10)
  ret void
}

define fastcc void @callee2(float %a0, float %a1, float %a2, float %a3,
                            float %a4, float %a5, float %a6, float %a7,
                            float %a8, float %a9, float %a10) {
entry:

; NOODDSPREG-LABEL:  callee2:

; Check that first 10 arguments are received in even float registers
; f0, f2, ... , f18. Check that 11th argument is received on stack.

; NOODDSPREG-DAG:    lw      $[[R0:[0-9]+]], %got(fa)(${{[0-9]+|gp}})
; NOODDSPREG-DAG:    swc1    $f0, 0($[[R0]])
; NOODDSPREG-DAG:    swc1    $f2, 4($[[R0]])
; NOODDSPREG-DAG:    swc1    $f4, 8($[[R0]])
; NOODDSPREG-DAG:    swc1    $f6, 12($[[R0]])
; NOODDSPREG-DAG:    swc1    $f8, 16($[[R0]])
; NOODDSPREG-DAG:    swc1    $f10, 20($[[R0]])
; NOODDSPREG-DAG:    swc1    $f12, 24($[[R0]])
; NOODDSPREG-DAG:    swc1    $f14, 28($[[R0]])
; NOODDSPREG-DAG:    swc1    $f16, 32($[[R0]])
; NOODDSPREG-DAG:    swc1    $f18, 36($[[R0]])

; NOODDSPREG-DAG:    lwc1    $[[F0:f[0-9]*[02468]]], {{[0-9]+}}($sp)
; NOODDSPREG-DAG:    swc1    $[[F0]], 40($[[R0]])

  store float %a0, ptr @fa, align 4
  store float %a1, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 1), align 4
  store float %a2, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 2), align 4
  store float %a3, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 3), align 4
  store float %a4, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 4), align 4
  store float %a5, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 5), align 4
  store float %a6, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 6), align 4
  store float %a7, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 7), align 4
  store float %a8, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 8), align 4
  store float %a9, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 9), align 4
  store float %a10, ptr getelementptr ([11 x float], ptr @fa, i32 0, i32 10), align 4
  ret void
}

define void @caller3() {
entry:

; FP64-NOODDSPREG-LABEL:  caller3:

; Check that first 10 arguments are passed in even float registers
; f0, f2, ... , f18. Check that 11th argument is passed on stack.

; FP64-NOODDSPREG-DAG:    lw      $[[R0:[0-9]+]], %got(da)(${{[0-9]+|gp}})
; FP64-NOODDSPREG-DAG:    ldc1    $f0, 0($[[R0]])
; FP64-NOODDSPREG-DAG:    ldc1    $f2, 8($[[R0]])
; FP64-NOODDSPREG-DAG:    ldc1    $f4, 16($[[R0]])
; FP64-NOODDSPREG-DAG:    ldc1    $f6, 24($[[R0]])
; FP64-NOODDSPREG-DAG:    ldc1    $f8, 32($[[R0]])
; FP64-NOODDSPREG-DAG:    ldc1    $f10, 40($[[R0]])
; FP64-NOODDSPREG-DAG:    ldc1    $f12, 48($[[R0]])
; FP64-NOODDSPREG-DAG:    ldc1    $f14, 56($[[R0]])
; FP64-NOODDSPREG-DAG:    ldc1    $f16, 64($[[R0]])
; FP64-NOODDSPREG-DAG:    ldc1    $f18, 72($[[R0]])

; FP64-NOODDSPREG-DAG:    ldc1    $[[F0:f[0-9]*[02468]]], 80($[[R0]])
; FP64-NOODDSPREG-DAG:    sdc1    $[[F0]], 0($sp)

  %0 = load double, ptr @da, align 8
  %1 = load double, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 1), align 8
  %2 = load double, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 2), align 8
  %3 = load double, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 3), align 8
  %4 = load double, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 4), align 8
  %5 = load double, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 5), align 8
  %6 = load double, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 6), align 8
  %7 = load double, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 7), align 8
  %8 = load double, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 8), align 8
  %9 = load double, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 9), align 8
  %10 = load double, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 10), align 8
  tail call fastcc void @callee3(double %0, double %1, double %2, double %3,
                                 double %4, double %5, double %6, double %7,
                                 double %8, double %9, double %10)
  ret void
}

define fastcc void @callee3(double %a0, double %a1, double %a2, double %a3,
                            double %a4, double %a5, double %a6, double %a7,
                            double %a8, double %a9, double %a10) {
entry:

; FP64-NOODDSPREG-LABEL:  callee3:


; Check that first 10 arguments are received in even float registers
; f0, f2, ... , f18. Check that 11th argument is received on stack.

; FP64-NOODDSPREG-DAG:    lw      $[[R0:[0-9]+]], %got(da)(${{[0-9]+|gp}})
; FP64-NOODDSPREG-DAG:    sdc1    $f0, 0($[[R0]])
; FP64-NOODDSPREG-DAG:    sdc1    $f2, 8($[[R0]])
; FP64-NOODDSPREG-DAG:    sdc1    $f4, 16($[[R0]])
; FP64-NOODDSPREG-DAG:    sdc1    $f6, 24($[[R0]])
; FP64-NOODDSPREG-DAG:    sdc1    $f8, 32($[[R0]])
; FP64-NOODDSPREG-DAG:    sdc1    $f10, 40($[[R0]])
; FP64-NOODDSPREG-DAG:    sdc1    $f12, 48($[[R0]])
; FP64-NOODDSPREG-DAG:    sdc1    $f14, 56($[[R0]])
; FP64-NOODDSPREG-DAG:    sdc1    $f16, 64($[[R0]])
; FP64-NOODDSPREG-DAG:    sdc1    $f18, 72($[[R0]])

; FP64-NOODDSPREG-DAG:    ldc1    $[[F0:f[0-9]*[02468]]], 0($sp)
; FP64-NOODDSPREG-DAG:    sdc1    $[[F0]], 80($[[R0]])

  store double %a0, ptr @da, align 8
  store double %a1, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 1), align 8
  store double %a2, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 2), align 8
  store double %a3, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 3), align 8
  store double %a4, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 4), align 8
  store double %a5, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 5), align 8
  store double %a6, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 6), align 8
  store double %a7, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 7), align 8
  store double %a8, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 8), align 8
  store double %a9, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 9), align 8
  store double %a10, ptr getelementptr ([11 x double], ptr @da, i32 0, i32 10), align 8
  ret void
}
