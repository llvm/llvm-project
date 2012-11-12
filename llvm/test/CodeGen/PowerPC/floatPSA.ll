; RUN: llc -O0 -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck %s

; This verifies that single-precision floating point values that can't
; be passed in registers are stored in the rightmost word of the parameter
; save area slot.  There are 13 architected floating-point registers, so
; the 14th is passed in storage.  The address of the 14th argument is
; 48 (fixed size of the linkage area) + 13 * 8 (first 13 args) + 4
; (offset to second word) = 156.

define float @bar(float %a, float %b, float %c, float %d, float %e, float %f, float %g, float %h, float %i, float %j, float %k, float %l, float %m, float %n) nounwind {
entry:
  %a.addr = alloca float, align 4
  %b.addr = alloca float, align 4
  %c.addr = alloca float, align 4
  %d.addr = alloca float, align 4
  %e.addr = alloca float, align 4
  %f.addr = alloca float, align 4
  %g.addr = alloca float, align 4
  %h.addr = alloca float, align 4
  %i.addr = alloca float, align 4
  %j.addr = alloca float, align 4
  %k.addr = alloca float, align 4
  %l.addr = alloca float, align 4
  %m.addr = alloca float, align 4
  %n.addr = alloca float, align 4
  store float %a, float* %a.addr, align 4
  store float %b, float* %b.addr, align 4
  store float %c, float* %c.addr, align 4
  store float %d, float* %d.addr, align 4
  store float %e, float* %e.addr, align 4
  store float %f, float* %f.addr, align 4
  store float %g, float* %g.addr, align 4
  store float %h, float* %h.addr, align 4
  store float %i, float* %i.addr, align 4
  store float %j, float* %j.addr, align 4
  store float %k, float* %k.addr, align 4
  store float %l, float* %l.addr, align 4
  store float %m, float* %m.addr, align 4
  store float %n, float* %n.addr, align 4
  %0 = load float* %n.addr, align 4
  ret float %0
}

; CHECK: lfs {{[0-9]+}}, 156(1)

define float @foo() nounwind {
entry:
  %a = alloca float, align 4
  %b = alloca float, align 4
  %c = alloca float, align 4
  %d = alloca float, align 4
  %e = alloca float, align 4
  %f = alloca float, align 4
  %g = alloca float, align 4
  %h = alloca float, align 4
  %i = alloca float, align 4
  %j = alloca float, align 4
  %k = alloca float, align 4
  %l = alloca float, align 4
  %m = alloca float, align 4
  %n = alloca float, align 4
  store float 1.000000e+00, float* %a, align 4
  store float 2.000000e+00, float* %b, align 4
  store float 3.000000e+00, float* %c, align 4
  store float 4.000000e+00, float* %d, align 4
  store float 5.000000e+00, float* %e, align 4
  store float 6.000000e+00, float* %f, align 4
  store float 7.000000e+00, float* %g, align 4
  store float 8.000000e+00, float* %h, align 4
  store float 9.000000e+00, float* %i, align 4
  store float 1.000000e+01, float* %j, align 4
  store float 1.100000e+01, float* %k, align 4
  store float 1.200000e+01, float* %l, align 4
  store float 1.300000e+01, float* %m, align 4
  store float 1.400000e+01, float* %n, align 4
  %0 = load float* %a, align 4
  %1 = load float* %b, align 4
  %2 = load float* %c, align 4
  %3 = load float* %d, align 4
  %4 = load float* %e, align 4
  %5 = load float* %f, align 4
  %6 = load float* %g, align 4
  %7 = load float* %h, align 4
  %8 = load float* %i, align 4
  %9 = load float* %j, align 4
  %10 = load float* %k, align 4
  %11 = load float* %l, align 4
  %12 = load float* %m, align 4
  %13 = load float* %n, align 4
  %call = call float @bar(float %0, float %1, float %2, float %3, float %4, float %5, float %6, float %7, float %8, float %9, float %10, float %11, float %12, float %13)
  ret float %call
}

; Note that stw is used instead of stfs because the value is a simple
; constant that can be created with a load-immediate in a GPR.
; CHECK: stw {{[0-9]+}}, 156(1)

