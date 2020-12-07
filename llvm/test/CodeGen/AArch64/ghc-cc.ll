; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s

; Check the GHC call convention works (aarch64)

@base  = external dso_local global i64 ; assigned to register: r19
@sp    = external dso_local global i64 ; assigned to register: r20
@hp    = external dso_local global i64 ; assigned to register: r21
@r1    = external dso_local global i64 ; assigned to register: r22
@r2    = external dso_local global i64 ; assigned to register: r23
@r3    = external dso_local global i64 ; assigned to register: r24
@r4    = external dso_local global i64 ; assigned to register: r25
@r5    = external dso_local global i64 ; assigned to register: r26
@r6    = external dso_local global i64 ; assigned to register: r27
@splim = external dso_local global i64 ; assigned to register: r28

@f1 = external dso_local global float  ; assigned to register: s8
@f2 = external dso_local global float  ; assigned to register: s9
@f3 = external dso_local global float  ; assigned to register: s10
@f4 = external dso_local global float  ; assigned to register: s11

@d1 = external dso_local global double ; assigned to register: d12
@d2 = external dso_local global double ; assigned to register: d13
@d3 = external dso_local global double ; assigned to register: d14
@d4 = external dso_local global double ; assigned to register: d15

define ghccc i64 @addtwo(i64 %x, i64 %y) nounwind {
entry:
  ; CHECK-LABEL: addtwo
  ; CHECK:       add      x0, x19, x20
  ; CHECK-NEXT:  ret
  %0 = add i64 %x, %y
  ret i64 %0
}

define void @zap(i64 %a, i64 %b) nounwind {
entry:
  ; CHECK-LABEL: zap
  ; CHECK-NOT:   mov   {{x[0-9]+}}, sp
  ; CHECK:       bl    addtwo
  ; CHECK-NEXT:  bl    foo
  %0 = call ghccc i64 @addtwo(i64 %a, i64 %b)
  call void @foo() nounwind
  ret void
}

define ghccc void @foo_i64 () nounwind {
entry:
  ; CHECK-LABEL: foo_i64
  ; CHECK:       adrp    {{x[0-9]+}}, base
  ; CHECK-NEXT:  ldr     x19, [{{x[0-9]+}}, :lo12:base]
  ; CHECK-NEXT:  bl      bar_i64
  ; CHECK-NEXT:  ret

  %0 = load i64, i64* @base
  tail call ghccc void @bar_i64( i64 %0 ) nounwind
  ret void
}

define ghccc void @foo_float () nounwind {
entry:
  ; CHECK-LABEL: foo_float
  ; CHECK:       adrp    {{x[0-9]+}}, f1
  ; CHECK-NEXT:  ldr     s8, [{{x[0-9]+}}, :lo12:f1]
  ; CHECK-NEXT:  bl      bar_float
  ; CHECK-NEXT:  ret

  %0 = load float, float* @f1
  tail call ghccc void @bar_float( float %0 ) nounwind
  ret void
}

define ghccc void @foo_double () nounwind {
entry:
  ; CHECK-LABEL: foo_double
  ; CHECK:       adrp    {{x[0-9]+}}, d1
  ; CHECK-NEXT:  ldr     d12, [{{x[0-9]+}}, :lo12:d1]
  ; CHECK-NEXT:  bl      bar_double
  ; CHECK-NEXT:  ret

  %0 = load double, double* @d1
  tail call ghccc void @bar_double( double %0 ) nounwind
  ret void
}

declare ghccc void @foo ()

declare ghccc void @bar_i64 (i64)
declare ghccc void @bar_float (float)
declare ghccc void @bar_double (double)
