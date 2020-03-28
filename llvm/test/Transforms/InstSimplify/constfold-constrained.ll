; RUN: opt < %s -instsimplify -S | FileCheck %s


; Verify that floor(10.1) is folded to 10.0 when the exception behavior is 'ignore'.
define double @floor_01() #0 {
entry:
  %result = call double @llvm.experimental.constrained.floor.f64(
                                               double 1.010000e+01,
                                               metadata !"fpexcept.ignore") #0
  ret double %result
  ; CHECK-LABEL: @floor_01
  ; CHECK: ret double 1.000000e+01
}

; Verify that floor(-10.1) is folded to -11.0 when the exception behavior is not 'ignore'.
define double @floor_02() #0 {
entry:
  %result = call double @llvm.experimental.constrained.floor.f64(
                                               double -1.010000e+01,
                                               metadata !"fpexcept.strict") #0
  ret double %result
  ; CHECK-LABEL: @floor_02
  ; CHECK: ret double -1.100000e+01
}

; Verify that ceil(10.1) is folded to 11.0 when the exception behavior is 'ignore'.
define double @ceil_01() #0 {
entry:
  %result = call double @llvm.experimental.constrained.ceil.f64(
                                               double 1.010000e+01,
                                               metadata !"fpexcept.ignore") #0
  ret double %result
  ; CHECK-LABEL: @ceil_01
  ; CHECK: ret double 1.100000e+01
}

; Verify that ceil(-10.1) is folded to -10.0 when the exception behavior is not 'ignore'.
define double @ceil_02() #0 {
entry:
  %result = call double @llvm.experimental.constrained.ceil.f64(
                                               double -1.010000e+01,
                                               metadata !"fpexcept.strict") #0
  ret double %result
  ; CHECK-LABEL: @ceil_02
  ; CHECK: ret double -1.000000e+01
}

; Verify that trunc(10.1) is folded to 10.0 when the exception behavior is 'ignore'.
define double @trunc_01() #0 {
entry:
  %result = call double @llvm.experimental.constrained.trunc.f64(
                                               double 1.010000e+01,
                                               metadata !"fpexcept.ignore") #0
  ret double %result
  ; CHECK-LABEL: @trunc_01
  ; CHECK: ret double 1.000000e+01
}

; Verify that trunc(-10.1) is folded to -10.0 when the exception behavior is NOT 'ignore'.
define double @trunc_02() #0 {
entry:
  %result = call double @llvm.experimental.constrained.trunc.f64(
                                               double -1.010000e+01,
                                               metadata !"fpexcept.strict") #0
  ret double %result
  ; CHECK-LABEL: @trunc_02
  ; CHECK: ret double -1.000000e+01
}

; Verify that round(10.5) is folded to 11.0 when the exception behavior is 'ignore'.
define double @round_01() #0 {
entry:
  %result = call double @llvm.experimental.constrained.round.f64(
                                               double 1.050000e+01,
                                               metadata !"fpexcept.ignore") #0
  ret double %result
  ; CHECK-LABEL: @round_01
  ; CHECK: ret double 1.100000e+01
}

; Verify that floor(-10.5) is folded to -11.0 when the exception behavior is NOT 'ignore'.
define double @round_02() #0 {
entry:
  %result = call double @llvm.experimental.constrained.round.f64(
                                               double -1.050000e+01,
                                               metadata !"fpexcept.strict") #0
  ret double %result
  ; CHECK-LABEL: @round_02
  ; CHECK: ret double -1.100000e+01
}

; Verify that nearbyint(10.5) is folded to 11.0 when the rounding mode is 'upward'.
define double @nearbyint_01() #0 {
entry:
  %result = call double @llvm.experimental.constrained.nearbyint.f64(
                                               double 1.050000e+01,
                                               metadata !"round.upward",
                                               metadata !"fpexcept.ignore") #0
  ret double %result
  ; CHECK-LABEL: @nearbyint_01
  ; CHECK: ret double 1.100000e+01
}

; Verify that nearbyint(10.5) is folded to 10.0 when the rounding mode is 'downward'.
define double @nearbyint_02() #0 {
entry:
  %result = call double @llvm.experimental.constrained.nearbyint.f64(
                                               double 1.050000e+01,
                                               metadata !"round.downward",
                                               metadata !"fpexcept.maytrap") #0
  ret double %result
  ; CHECK-LABEL: @nearbyint_02
  ; CHECK: ret double 1.000000e+01
}

; Verify that nearbyint(10.5) is folded to 10.0 when the rounding mode is 'towardzero'.
define double @nearbyint_03() #0 {
entry:
  %result = call double @llvm.experimental.constrained.nearbyint.f64(
                                               double 1.050000e+01,
                                               metadata !"round.towardzero",
                                               metadata !"fpexcept.strict") #0
  ret double %result
  ; CHECK-LABEL: @nearbyint_03
  ; CHECK: ret double 1.000000e+01
}

; Verify that nearbyint(10.5) is folded to 10.0 when the rounding mode is 'tonearest'.
define double @nearbyint_04() #0 {
entry:
  %result = call double @llvm.experimental.constrained.nearbyint.f64(
                                               double 1.050000e+01,
                                               metadata !"round.tonearest",
                                               metadata !"fpexcept.strict") #0
  ret double %result
  ; CHECK-LABEL: @nearbyint_04
  ; CHECK: ret double 1.000000e+01
}

; Verify that nearbyint(10.5) is NOT folded if the rounding mode is 'dynamic'.
define double @nearbyint_05() #0 {
entry:
  %result = call double @llvm.experimental.constrained.nearbyint.f64(
                                               double 1.050000e+01,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %result
  ; CHECK-LABEL: @nearbyint_05
  ; CHECK: [[VAL:%.+]] = {{.*}}call double @llvm.experimental.constrained.nearbyint
  ; CHECK: ret double [[VAL]]
}

; Verify that trunc(SNAN) is NOT folded if the exception behavior mode is not 'ignore'.
define double @nonfinite_01() #0 {
entry:
  %result = call double @llvm.experimental.constrained.trunc.f64(
                                               double 0x7ff4000000000000,
                                               metadata !"fpexcept.strict") #0
  ret double %result
  ; CHECK-LABEL: @nonfinite_01
  ; CHECK: [[VAL:%.+]] = {{.*}}call double @llvm.experimental.constrained.trunc
  ; CHECK: ret double [[VAL]]
}

; Verify that trunc(SNAN) is folded to QNAN if the exception behavior mode is 'ignore'.
define double @nonfinite_02() #0 {
entry:
  %result = call double @llvm.experimental.constrained.trunc.f64(
                                               double 0x7ff4000000000000,
                                               metadata !"fpexcept.ignore") #0
  ret double %result
  ; CHECK-LABEL: @nonfinite_02
  ; CHECK: ret double 0x7FF8000000000000
}

; Verify that trunc(QNAN) is folded even if the exception behavior mode is not 'ignore'.
define double @nonfinite_03() #0 {
entry:
  %result = call double @llvm.experimental.constrained.trunc.f64(
                                               double 0x7ff8000000000000,
                                               metadata !"fpexcept.strict") #0
  ret double %result
  ; CHECK-LABEL: @nonfinite_03
  ; CHECK: ret double 0x7FF8000000000000
}

; Verify that trunc(+Inf) is folded even if the exception behavior mode is not 'ignore'.
define double @nonfinite_04() #0 {
entry:
  %result = call double @llvm.experimental.constrained.trunc.f64(
                                               double 0x7ff0000000000000,
                                               metadata !"fpexcept.strict") #0
  ret double %result
  ; CHECK-LABEL: @nonfinite_04
  ; CHECK: ret double 0x7FF0000000000000
}

; Verify that rint(10) is folded to 10.0 when the rounding mode is 'tonearest'.
define double @rint_01() #0 {
entry:
  %result = call double @llvm.experimental.constrained.rint.f64(
                                               double 1.000000e+01,
                                               metadata !"round.tonearest",
                                               metadata !"fpexcept.strict") #0
  ret double %result
  ; CHECK-LABEL: @rint_01
  ; CHECK: ret double 1.000000e+01
}

; Verify that rint(10.1) is NOT folded to 10.0 when the exception behavior is 'strict'.
define double @rint_02() #0 {
entry:
  %result = call double @llvm.experimental.constrained.rint.f64(
                                               double 1.010000e+01,
                                               metadata !"round.tonearest",
                                               metadata !"fpexcept.strict") #0
  ret double %result
  ; CHECK-LABEL: @rint_02
  ; CHECK: [[VAL:%.+]] = {{.*}}call double @llvm.experimental.constrained.rint
  ; CHECK: ret double [[VAL]]
}

; Verify that rint(10.1) is folded to 10.0 when the exception behavior is not 'strict'.
define double @rint_03() #0 {
entry:
  %result = call double @llvm.experimental.constrained.rint.f64(
                                               double 1.010000e+01,
                                               metadata !"round.tonearest",
                                               metadata !"fpexcept.maytrap") #0
  ret double %result
  ; CHECK-LABEL: @rint_03
  ; CHECK: ret double 1.000000e+01
}


attributes #0 = { strictfp }

declare double @llvm.experimental.constrained.nearbyint.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.floor.f64(double, metadata)
declare double @llvm.experimental.constrained.ceil.f64(double, metadata)
declare double @llvm.experimental.constrained.trunc.f64(double, metadata)
declare double @llvm.experimental.constrained.round.f64(double, metadata)
declare double @llvm.experimental.constrained.rint.f64(double, metadata, metadata)

