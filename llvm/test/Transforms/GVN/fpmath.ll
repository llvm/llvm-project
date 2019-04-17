; RUN: opt -gvn -S < %s | FileCheck %s

define double @test1(double %x, double %y) {
; CHECK: @test1(double %x, double %y)
; CHECK: %add1 = fadd double %x, %y
; CHECK-NOT: fpmath
; CHECK: %foo = fadd double %add1, %add1
  %add1 = fadd double %x, %y, !fpmath !0
  %add2 = fadd double %x, %y
  %foo = fadd double %add1, %add2
  ret double %foo
}

define double @test2(double %x, double %y) {
; CHECK: @test2(double %x, double %y)
; CHECK: %add1 = fadd double %x, %y, !fpmath !0
; CHECK: %foo = fadd double %add1, %add1
  %add1 = fadd double %x, %y, !fpmath !0
  %add2 = fadd double %x, %y, !fpmath !0
  %foo = fadd double %add1, %add2
  ret double %foo
}

define double @test3(double %x, double %y) {
; CHECK: @test3(double %x, double %y)
; CHECK: %add1 = fadd double %x, %y, !fpmath !1
; CHECK: %foo = fadd double %add1, %add1
  %add1 = fadd double %x, %y, !fpmath !1
  %add2 = fadd double %x, %y, !fpmath !0
  %foo = fadd double %add1, %add2
  ret double %foo
}

define double @test4(double %x, double %y) {
; CHECK: @test4(double %x, double %y)
; CHECK: %add1 = fadd double %x, %y, !fpmath !1
; CHECK: %foo = fadd double %add1, %add1
  %add1 = fadd double %x, %y, !fpmath !0
  %add2 = fadd double %x, %y, !fpmath !1
  %foo = fadd double %add1, %add2
  ret double %foo
}

!0 = !{ float 5.0 }
!1 = !{ float 2.5 }
