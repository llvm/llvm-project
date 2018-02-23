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
