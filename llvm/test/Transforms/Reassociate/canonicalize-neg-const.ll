; RUN: opt -reassociate -gvn -S < %s | FileCheck %s

; (x + 0.1234 * y) * (x + -0.1234 * y) -> (x + 0.1234 * y) * (x - 0.1234 * y)
define double @test1(double %x, double %y) {
; CHECK-LABEL: @test1
; CHECK-NEXT: fmul double %y, 1.234000e-01
; CHECK-NEXT: fadd double %x, %mul
; CHECK-NEXT: fsub double %x, %mul
; CHECK-NEXT: fmul double %add{{.*}}, %add{{.*}}
; CHECK-NEXT: ret double %mul

  %mul = fmul double 1.234000e-01, %y
  %add = fadd double %mul, %x
  %mul1 = fmul double -1.234000e-01, %y
  %add2 = fadd double %mul1, %x
  %mul3 = fmul double %add, %add2
  ret double %mul3
}

; (x + -0.1234 * y) * (x + -0.1234 * y) -> (x - 0.1234 * y) * (x - 0.1234 * y)
define double @test2(double %x, double %y) {
; CHECK-LABEL: @test2
; CHECK-NEXT: fmul double %y, 1.234000e-01
; CHECK-NEXT: fsub double %x, %mul
; CHECK-NEXT: fmul double %add{{.*}}, %add{{.*}}
; CHECK-NEXT: ret double %mul

  %mul = fmul double %y, -1.234000e-01
  %add = fadd double %mul, %x
  %mul1 = fmul double %y, -1.234000e-01
  %add2 = fadd double %mul1, %x
  %mul3 = fmul double %add, %add2
  ret double %mul3
}

; (x + 0.1234 * y) * (x - -0.1234 * y) -> (x + 0.1234 * y) * (x + 0.1234 * y)
define double @test3(double %x, double %y) {
; CHECK-LABEL: @test3
; CHECK-NEXT: fmul double %y, 1.234000e-01
; CHECK-NEXT: fadd double %x, %mul
; CHECK-NEXT: fmul double %add{{.*}}, %add{{.*}}
; CHECK-NEXT: ret double

  %mul = fmul double %y, 1.234000e-01
  %add = fadd double %mul, %x
  %mul1 = fmul double %y, -1.234000e-01
  %add2 = fsub double %x, %mul1
  %mul3 = fmul double %add, %add2
  ret double %mul3
}

; Canonicalize (x - -0.1234 * y)
define double @test5(double %x, double %y) {
; CHECK-LABEL: @test5
; CHECK-NEXT: fmul double %y, 1.234000e-01
; CHECK-NEXT: fadd double %x, %mul
; CHECK-NEXT: ret double

  %mul = fmul double -1.234000e-01, %y
  %sub = fsub double %x, %mul
  ret double %sub
}

; Don't modify (-0.1234 * y - x)
define double @test6(double %x, double %y) {
; CHECK-LABEL: @test6
; CHECK-NEXT: fmul double %y, -1.234000e-01
; CHECK-NEXT: fsub double %mul, %x
; CHECK-NEXT: ret double %sub

  %mul = fmul double -1.234000e-01, %y
  %sub = fsub double %mul, %x
  ret double %sub
}

; Canonicalize (-0.1234 * y + x) -> (x - 0.1234 * y)
define double @test7(double %x, double %y) {
; CHECK-LABEL: @test7
; CHECK-NEXT: fmul double %y, 1.234000e-01
; CHECK-NEXT: fsub double %x, %mul
; CHECK-NEXT: ret double %add

  %mul = fmul double -1.234000e-01, %y
  %add = fadd double %mul, %x
  ret double %add
}

; Canonicalize (y * -0.1234 + x) -> (x - 0.1234 * y)
define double @test8(double %x, double %y) {
; CHECK-LABEL: @test8
; CHECK-NEXT: fmul double %y, 1.234000e-01
; CHECK-NEXT: fsub double %x, %mul
; CHECK-NEXT: ret double %add

  %mul = fmul double %y, -1.234000e-01
  %add = fadd double %mul, %x
  ret double %add
}

; Canonicalize (x - -0.1234 / y)
define double @test9(double %x, double %y) {
; CHECK-LABEL: @test9
; CHECK-NEXT: fdiv double 1.234000e-01, %y
; CHECK-NEXT: fadd double %x, %div
; CHECK-NEXT: ret double

  %div = fdiv double -1.234000e-01, %y
  %sub = fsub double %x, %div
  ret double %sub
}

; Don't modify (-0.1234 / y - x)
define double @test10(double %x, double %y) {
; CHECK-LABEL: @test10
; CHECK-NEXT: fdiv double -1.234000e-01, %y
; CHECK-NEXT: fsub double %div, %x
; CHECK-NEXT: ret double %sub

  %div = fdiv double -1.234000e-01, %y
  %sub = fsub double %div, %x
  ret double %sub
}

; Canonicalize (-0.1234 / y + x) -> (x - 0.1234 / y)
define double @test11(double %x, double %y) {
; CHECK-LABEL: @test11
; CHECK-NEXT: fdiv double 1.234000e-01, %y
; CHECK-NEXT: fsub double %x, %div
; CHECK-NEXT: ret double %add

  %div = fdiv double -1.234000e-01, %y
  %add = fadd double %div, %x
  ret double %add
}

; Canonicalize (y / -0.1234 + x) -> (x - y / 0.1234)
define double @test12(double %x, double %y) {
; CHECK-LABEL: @test12
; CHECK-NEXT: fdiv double %y, 1.234000e-01
; CHECK-NEXT: fsub double %x, %div
; CHECK-NEXT: ret double %add

  %div = fdiv double %y, -1.234000e-01
  %add = fadd double %div, %x
  ret double %add
}

; Don't create an NSW violation
define i4 @test13(i4 %x) {
; CHECK-LABEL: @test13
; CHECK-NEXT: %[[mul:.*]] = mul nsw i4 %x, -2
; CHECK-NEXT: %[[add:.*]] = add i4 %[[mul]], 3
  %mul = mul nsw i4 %x, -2
  %add = add i4 %mul, 3
  ret i4 %add
}

; This tests used to cause an infinite loop where we would loop between
; canonicalizing the negated constant (i.e., (X + Y*-5.0) -> (X - Y*5.0)) and
; breaking up a subtract (i.e., (X - Y*5.0) -> X + (0 - Y*5.0)). To break the
; cycle, we don't canonicalize the negative constant if we're going to later
; break up the subtract.
;
; Check to make sure we don't canonicalize
;   (%pow2*-5.0 + %sub) -> (%sub - %pow2*5.0)
; as we would later break up this subtract causing a cycle.
;
; CHECK-LABEL: @pr34078
; CHECK: %mul5.neg = fmul fast double %pow2, -5.000000e-01
; CHECK: %sub1 = fadd fast double %mul5.neg, %sub
define double @pr34078(double %A) {
  %sub = fsub fast double 1.000000e+00, %A
  %pow2 = fmul double %A, %A
  %mul5 = fmul fast double %pow2, 5.000000e-01
  %sub1 = fsub fast double %sub, %mul5
  %add = fadd fast double %sub1, %sub1
  ret double %add
}
