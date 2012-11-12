; RUN: opt < %s -reassociate -S | FileCheck %s

define i32 @test1(i32 %a, i32 %b) {
; CHECK: @test1
; CHECK: mul i32 %a, %a
; CHECK-NEXT: mul i32 %a, 2
; CHECK-NEXT: add
; CHECK-NEXT: mul
; CHECK-NEXT: add
; CHECK-NEXT: ret

entry:
	%tmp.2 = mul i32 %a, %a
	%tmp.5 = shl i32 %a, 1
	%tmp.6 = mul i32 %tmp.5, %b
	%tmp.10 = mul i32 %b, %b
	%tmp.7 = add i32 %tmp.6, %tmp.2
	%tmp.11 = add i32 %tmp.7, %tmp.10
	ret i32 %tmp.11
}

define i32 @test2(i32 %t) {
; CHECK: @test2
; CHECK: mul
; CHECK-NEXT: add
; CHECK-NEXT: ret

entry:
	%a = mul i32 %t, 6
	%b = mul i32 %t, 36
	%c = add i32 %b, 15
	%d = add i32 %c, %a
	ret i32 %d
}

define i32 @test3(i32 %x) {
; (x^8)
; CHECK: @test3
; CHECK: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: ret

entry:
  %a = mul i32 %x, %x
  %b = mul i32 %a, %x
  %c = mul i32 %b, %x
  %d = mul i32 %c, %x
  %e = mul i32 %d, %x
  %f = mul i32 %e, %x
  %g = mul i32 %f, %x
  ret i32 %g
}

define i32 @test4(i32 %x) {
; (x^7)
; CHECK: @test4
; CHECK: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: ret

entry:
  %a = mul i32 %x, %x
  %b = mul i32 %a, %x
  %c = mul i32 %b, %x
  %d = mul i32 %c, %x
  %e = mul i32 %d, %x
  %f = mul i32 %e, %x
  ret i32 %f
}

define i32 @test5(i32 %x, i32 %y) {
; (x^4) * (y^2)
; CHECK: @test5
; CHECK: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: ret

entry:
  %a = mul i32 %x, %y
  %b = mul i32 %a, %y
  %c = mul i32 %b, %x
  %d = mul i32 %c, %x
  %e = mul i32 %d, %x
  ret i32 %e
}

define i32 @test6(i32 %x, i32 %y, i32 %z) {
; (x^5) * (y^3) * z
; CHECK: @test6
; CHECK: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: ret

entry:
  %a = mul i32 %x, %y
  %b = mul i32 %a, %x
  %c = mul i32 %b, %y
  %d = mul i32 %c, %x
  %e = mul i32 %d, %y
  %f = mul i32 %e, %x
  %g = mul i32 %f, %z
  %h = mul i32 %g, %x
  ret i32 %h
}

define i32 @test7(i32 %x, i32 %y, i32 %z) {
; (x^4) * (y^3) * (z^2)
; CHECK: @test7
; CHECK: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: mul
; CHECK-NEXT: ret

entry:
  %a = mul i32 %y, %x
  %b = mul i32 %a, %z
  %c = mul i32 %b, %z
  %d = mul i32 %c, %x
  %e = mul i32 %d, %y
  %f = mul i32 %e, %y
  %g = mul i32 %f, %x
  %h = mul i32 %g, %x
  ret i32 %h
}
