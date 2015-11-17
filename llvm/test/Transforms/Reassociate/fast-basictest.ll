; RUN: opt < %s -reassociate -gvn -instcombine -S | FileCheck %s

; With reassociation, constant folding can eliminate the 12 and -12 constants.
define float @test1(float %arg) {
; CHECK-LABEL: @test1
; CHECK-NEXT: fsub fast float -0.000000e+00, %arg
; CHECK-NEXT: ret float

  %tmp1 = fsub fast float -1.200000e+01, %arg
  %tmp2 = fadd fast float %tmp1, 1.200000e+01
  ret float %tmp2
}

define float @test2(float %reg109, float %reg1111) {
; CHECK-LABEL: @test2
; CHECK-NEXT: fadd float %reg109, -3.000000e+01
; CHECK-NEXT: fadd float %reg115, %reg1111
; CHECK-NEXT: fadd float %reg116, 3.000000e+01
; CHECK-NEXT: ret float

  %reg115 = fadd float %reg109, -3.000000e+01
  %reg116 = fadd float %reg115, %reg1111
  %reg117 = fadd float %reg116, 3.000000e+01
  ret float %reg117
}

define float @test3(float %reg109, float %reg1111) {
; CHECK-LABEL: @test3
; CHECK-NEXT: %reg117 = fadd fast float %reg109, %reg1111
; CHECK-NEXT:  ret float %reg117

  %reg115 = fadd fast float %reg109, -3.000000e+01
  %reg116 = fadd fast float %reg115, %reg1111
  %reg117 = fadd fast float %reg116, 3.000000e+01
  ret float %reg117
}

@fe = external global float
@fa = external global float
@fb = external global float
@fc = external global float
@ff = external global float

define void @test4() {
; CHECK-LABEL: @test4
; CHECK: fadd fast float
; CHECK: fadd fast float
; CHECK-NOT: fadd fast float
; CHECK: ret void

  %A = load float, float* @fa
  %B = load float, float* @fb
  %C = load float, float* @fc
  %t1 = fadd fast float %A, %B
  %t2 = fadd fast float %t1, %C
  %t3 = fadd fast float %C, %A
  %t4 = fadd fast float %t3, %B
  ; e = (a+b)+c;
  store float %t2, float* @fe
  ; f = (a+c)+b
  store float %t4, float* @ff
  ret void
}

define void @test5() {
; CHECK-LABEL: @test5
; CHECK: fadd fast float
; CHECK: fadd fast float
; CHECK-NOT: fadd
; CHECK: ret void

  %A = load float, float* @fa
  %B = load float, float* @fb
  %C = load float, float* @fc
  %t1 = fadd fast float %A, %B
  %t2 = fadd fast float %t1, %C
  %t3 = fadd fast float %C, %A
  %t4 = fadd fast float %t3, %B
  ; e = c+(a+b)
  store float %t2, float* @fe
  ; f = (c+a)+b
  store float %t4, float* @ff
  ret void
}

define void @test6() {
; CHECK-LABEL: @test6
; CHECK: fadd fast float
; CHECK: fadd fast float
; CHECK-NOT: fadd
; CHECK: ret void

  %A = load float, float* @fa
  %B = load float, float* @fb
  %C = load float, float* @fc
  %t1 = fadd fast float %B, %A
  %t2 = fadd fast float %t1, %C
  %t3 = fadd fast float %C, %A
  %t4 = fadd fast float %t3, %B
  ; e = c+(b+a)
  store float %t2, float* @fe
  ; f = (c+a)+b
  store float %t4, float* @ff
  ret void
}

define float @test7(float %A, float %B, float %C) {
; CHECK-LABEL: @test7
; CHECK-NEXT: fadd fast float %C, %B
; CHECK-NEXT: fmul fast float %A, %A
; CHECK-NEXT: fmul fast float %tmp3, %tmp2
; CHECK-NEXT: ret float

  %aa = fmul fast float %A, %A
  %aab = fmul fast float %aa, %B
  %ac = fmul fast float %A, %C
  %aac = fmul fast float %ac, %A
  %r = fadd fast float %aab, %aac
  ret float %r
}

define float @test8(float %X, float %Y, float %Z) {
; CHECK-LABEL: @test8
; CHECK-NEXT: fmul fast float %Y, %X
; CHECK-NEXT: fsub fast float %Z
; CHECK-NEXT: ret float

  %A = fsub fast float 0.0, %X
  %B = fmul fast float %A, %Y
  ; (-X)*Y + Z -> Z-X*Y
  %C = fadd fast float %B, %Z
  ret float %C
}

define float @test9(float %X) {
; CHECK-LABEL: @test9
; CHECK-NEXT: fmul fast float %X, 9.400000e+01
; CHECK-NEXT: ret float

  %Y = fmul fast float %X, 4.700000e+01
  %Z = fadd fast float %Y, %Y
  ret float %Z
}

define float @test10(float %X) {
; CHECK-LABEL: @test10
; CHECK-NEXT: fmul fast float %X, 3.000000e+00
; CHECK-NEXT: ret float

  %Y = fadd fast float %X ,%X
  %Z = fadd fast float %Y, %X
  ret float %Z
}

define float @test11(float %W) {
; CHECK-LABEL: test11
; CHECK-NEXT: fmul fast float %W, 3.810000e+02
; CHECK-NEXT: ret float

  %X = fmul fast float %W, 127.0
  %Y = fadd fast float %X ,%X
  %Z = fadd fast float %Y, %X
  ret float %Z
}

define float @test12(float %X) {
; CHECK-LABEL: @test12
; CHECK-NEXT: fmul fast float %X, -3.000000e+00
; CHECK-NEXT: fadd fast float %factor, 6.000000e+00
; CHECK-NEXT: ret float

  %A = fsub fast float 1.000000e+00, %X
  %B = fsub fast float 2.000000e+00, %X
  %C = fsub fast float 3.000000e+00, %X
  %Y = fadd fast float %A ,%B
  %Z = fadd fast float %Y, %C
  ret float %Z
}

define float @test13(float %X1, float %X2, float %X3) {
; CHECK-LABEL: @test13
; CHECK-NEXT: fsub fast float %X3, %X2
; CHECK-NEXT: fmul fast float {{.*}}, %X1
; CHECK-NEXT: ret float

  %A = fsub fast float 0.000000e+00, %X1
  %B = fmul fast float %A, %X2   ; -X1*X2
  %C = fmul fast float %X1, %X3  ; X1*X3
  %D = fadd fast float %B, %C    ; -X1*X2 + X1*X3 -> X1*(X3-X2)
  ret float %D
}

define float @test14(float %X1, float %X2) {
; CHECK-LABEL: @test14
; CHECK-NEXT: fsub fast float %X1, %X2
; CHECK-NEXT: fmul fast float %1, 4.700000e+01
; CHECK-NEXT: ret float

  %B = fmul fast float %X1, 47.   ; X1*47
  %C = fmul fast float %X2, -47.  ; X2*-47
  %D = fadd fast float %B, %C    ; X1*47 + X2*-47 -> 47*(X1-X2)
  ret float %D
}

define float @test15(float %arg) {
; CHECK-LABEL: test15
; CHECK-NEXT: fmul fast float %arg, 1.440000e+02
; CHECK-NEXT: ret float %tmp2

  %tmp1 = fmul fast float 1.200000e+01, %arg
  %tmp2 = fmul fast float %tmp1, 1.200000e+01
  ret float %tmp2
}

; (b+(a+1234))+-a -> b+1234
define float @test16(float %b, float %a) {
; CHECK-LABEL: @test16
; CHECK-NEXT: fadd fast float %b, 1.234000e+03
; CHECK-NEXT: ret float

  %1 = fadd fast float %a, 1234.0
  %2 = fadd fast float %b, %1
  %3 = fsub fast float 0.0, %a
  %4 = fadd fast float %2, %3
  ret float %4
}

; Test that we can turn things like X*-(Y*Z) -> X*-1*Y*Z.

define float @test17(float %a, float %b, float %z) {
; CHECK-LABEL: test17
; CHECK-NEXT: fmul fast float %a, 1.234500e+04
; CHECK-NEXT: fmul fast float %e, %b
; CHECK-NEXT: fmul fast float %f, %z
; CHECK-NEXT: ret float

  %c = fsub fast float 0.000000e+00, %z
  %d = fmul fast float %a, %b
  %e = fmul fast float %c, %d
  %f = fmul fast float %e, 1.234500e+04
  %g = fsub fast float 0.000000e+00, %f
  ret float %g
}

define float @test18(float %a, float %b, float %z) {
; CHECK-LABEL: test18
; CHECK-NEXT: fmul fast float %a, 4.000000e+01
; CHECK-NEXT: fmul fast float %e, %z
; CHECK-NEXT: ret float

  %d = fmul fast float %z, 4.000000e+01
  %c = fsub fast float 0.000000e+00, %d
  %e = fmul fast float %a, %c
  %f = fsub fast float 0.000000e+00, %e
  ret float %f
}

; With sub reassociation, constant folding can eliminate the 12 and -12 constants.
define float @test19(float %A, float %B) {
; CHECK-LABEL: @test19
; CHECK-NEXT: fsub fast float %A, %B
; CHECK-NEXT: ret float
  %X = fadd fast float -1.200000e+01, %A
  %Y = fsub fast float %X, %B
  %Z = fadd fast float %Y, 1.200000e+01
  ret float %Z
}

; With sub reassociation, constant folding can eliminate the uses of %a.
define float @test20(float %a, float %b, float %c) nounwind  {
; CHECK-LABEL: @test20
; CHECK-NEXT: fsub fast float -0.000000e+00, %b
; CHECK-NEXT: fsub fast float %b.neg, %c
; CHECK-NEXT: ret float

; FIXME: Should be able to generate the below, which may expose more
;        opportunites for FAdd reassociation.
; %sum = fadd fast float %c, %b
; %tmp7 = fsub fast float 0, %sum

  %tmp3 = fsub fast float %a, %b
  %tmp5 = fsub fast float %tmp3, %c
  %tmp7 = fsub fast float %tmp5, %a
  ret float %tmp7
}
