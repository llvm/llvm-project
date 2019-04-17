; RUN: opt -reassociate -gvn -S < %s | FileCheck %s

; (x + 0.1234 * y) * (x + -0.1234 * y) -> (x + 0.1234 * y) * (x - 0.1234 * y)
define double @test1(double %x, double %y) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:    [[MUL:%.*]] = fmul double %y, 1.234000e-01
; CHECK-NEXT:    [[ADD:%.*]] = fadd double %x, [[MUL]]
; CHECK-NEXT:    [[ADD21:%.*]] = fsub double %x, [[MUL]]
; CHECK-NEXT:    [[MUL3:%.*]] = fmul double [[ADD]], [[ADD21]]
; CHECK-NEXT:    ret double [[MUL3]]
;
  %mul = fmul double 1.234000e-01, %y
  %add = fadd double %mul, %x
  %mul1 = fmul double -1.234000e-01, %y
  %add2 = fadd double %mul1, %x
  %mul3 = fmul double %add, %add2
  ret double %mul3
}

; (x + -0.1234 * y) * (x + -0.1234 * y) -> (x - 0.1234 * y) * (x - 0.1234 * y)
define double @test2(double %x, double %y) {
; CHECK-LABEL: @test2(
; CHECK-NEXT:    [[MUL:%.*]] = fmul double %y, 1.234000e-01
; CHECK-NEXT:    [[ADD1:%.*]] = fsub double %x, [[MUL]]
; CHECK-NEXT:    [[MUL3:%.*]] = fmul double [[ADD1]], [[ADD1]]
; CHECK-NEXT:    ret double [[MUL3]]
;
  %mul = fmul double %y, -1.234000e-01
  %add = fadd double %mul, %x
  %mul1 = fmul double %y, -1.234000e-01
  %add2 = fadd double %mul1, %x
  %mul3 = fmul double %add, %add2
  ret double %mul3
}

; (x + 0.1234 * y) * (x - -0.1234 * y) -> (x + 0.1234 * y) * (x + 0.1234 * y)
define double @test3(double %x, double %y) {
; CHECK-LABEL: @test3(
; CHECK-NEXT:    [[MUL:%.*]] = fmul double %y, 1.234000e-01
; CHECK-NEXT:    [[ADD:%.*]] = fadd double %x, [[MUL]]
; CHECK-NEXT:    [[MUL3:%.*]] = fmul double [[ADD]], [[ADD]]
; CHECK-NEXT:    ret double [[MUL3]]
;
  %mul = fmul double %y, 1.234000e-01
  %add = fadd double %mul, %x
  %mul1 = fmul double %y, -1.234000e-01
  %add2 = fsub double %x, %mul1
  %mul3 = fmul double %add, %add2
  ret double %mul3
}

; Canonicalize (x - -0.1234 * y)
define double @test5(double %x, double %y) {
; CHECK-LABEL: @test5(
; CHECK-NEXT:    [[MUL:%.*]] = fmul double %y, 1.234000e-01
; CHECK-NEXT:    [[SUB1:%.*]] = fadd double %x, [[MUL]]
; CHECK-NEXT:    ret double [[SUB1]]
;
  %mul = fmul double -1.234000e-01, %y
  %sub = fsub double %x, %mul
  ret double %sub
}

; Don't modify (-0.1234 * y - x)
define double @test6(double %x, double %y) {
; CHECK-LABEL: @test6(
; CHECK-NEXT:    [[MUL:%.*]] = fmul double %y, -1.234000e-01
; CHECK-NEXT:    [[SUB:%.*]] = fsub double [[MUL]], %x
; CHECK-NEXT:    ret double [[SUB]]
;
  %mul = fmul double -1.234000e-01, %y
  %sub = fsub double %mul, %x
  ret double %sub
}

; Canonicalize (-0.1234 * y + x) -> (x - 0.1234 * y)
define double @test7(double %x, double %y) {
; CHECK-LABEL: @test7(
; CHECK-NEXT:    [[MUL:%.*]] = fmul double %y, 1.234000e-01
; CHECK-NEXT:    [[ADD1:%.*]] = fsub double %x, [[MUL]]
; CHECK-NEXT:    ret double [[ADD1]]
;
  %mul = fmul double -1.234000e-01, %y
  %add = fadd double %mul, %x
  ret double %add
}

; Canonicalize (y * -0.1234 + x) -> (x - 0.1234 * y)
define double @test8(double %x, double %y) {
; CHECK-LABEL: @test8(
; CHECK-NEXT:    [[MUL:%.*]] = fmul double %y, 1.234000e-01
; CHECK-NEXT:    [[ADD1:%.*]] = fsub double %x, [[MUL]]
; CHECK-NEXT:    ret double [[ADD1]]
;
  %mul = fmul double %y, -1.234000e-01
  %add = fadd double %mul, %x
  ret double %add
}

; Canonicalize (x - -0.1234 / y)
define double @test9(double %x, double %y) {
; CHECK-LABEL: @test9(
; CHECK-NEXT:    [[DIV:%.*]] = fdiv double 1.234000e-01, %y
; CHECK-NEXT:    [[SUB1:%.*]] = fadd double %x, [[DIV]]
; CHECK-NEXT:    ret double [[SUB1]]
;
  %div = fdiv double -1.234000e-01, %y
  %sub = fsub double %x, %div
  ret double %sub
}

; Don't modify (-0.1234 / y - x)
define double @test10(double %x, double %y) {
; CHECK-LABEL: @test10(
; CHECK-NEXT:    [[DIV:%.*]] = fdiv double -1.234000e-01, %y
; CHECK-NEXT:    [[SUB:%.*]] = fsub double [[DIV]], %x
; CHECK-NEXT:    ret double [[SUB]]
;
  %div = fdiv double -1.234000e-01, %y
  %sub = fsub double %div, %x
  ret double %sub
}

; Canonicalize (-0.1234 / y + x) -> (x - 0.1234 / y)
define double @test11(double %x, double %y) {
; CHECK-LABEL: @test11(
; CHECK-NEXT:    [[DIV:%.*]] = fdiv double 1.234000e-01, %y
; CHECK-NEXT:    [[ADD1:%.*]] = fsub double %x, [[DIV]]
; CHECK-NEXT:    ret double [[ADD1]]
;
  %div = fdiv double -1.234000e-01, %y
  %add = fadd double %div, %x
  ret double %add
}

; Canonicalize (y / -0.1234 + x) -> (x - y / 0.1234)
define double @test12(double %x, double %y) {
; CHECK-LABEL: @test12(
; CHECK-NEXT:    [[DIV:%.*]] = fdiv double %y, 1.234000e-01
; CHECK-NEXT:    [[ADD1:%.*]] = fsub double %x, [[DIV]]
; CHECK-NEXT:    ret double [[ADD1]]
;
  %div = fdiv double %y, -1.234000e-01
  %add = fadd double %div, %x
  ret double %add
}

; Don't create an NSW violation
define i4 @test13(i4 %x) {
; CHECK-LABEL: @test13(
; CHECK-NEXT:    [[MUL:%.*]] = mul nsw i4 %x, -2
; CHECK-NEXT:    [[ADD:%.*]] = add i4 [[MUL]], 3
; CHECK-NEXT:    ret i4 [[ADD]]
;
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

define double @pr34078(double %A) {
; CHECK-LABEL: @pr34078(
; CHECK-NEXT:    [[SUB:%.*]] = fsub fast double 1.000000e+00, %A
; CHECK-NEXT:    [[POW2:%.*]] = fmul double %A, %A
; CHECK-NEXT:    [[MUL5_NEG:%.*]] = fmul fast double [[POW2]], -5.000000e-01
; CHECK-NEXT:    [[SUB1:%.*]] = fadd fast double [[MUL5_NEG]], [[SUB]]
; CHECK-NEXT:    [[FACTOR:%.*]] = fmul fast double [[SUB1]], 2.000000e+00
; CHECK-NEXT:    ret double [[FACTOR]]
;
  %sub = fsub fast double 1.000000e+00, %A
  %pow2 = fmul double %A, %A
  %mul5 = fmul fast double %pow2, 5.000000e-01
  %sub1 = fsub fast double %sub, %mul5
  %add = fadd fast double %sub1, %sub1
  ret double %add
}
