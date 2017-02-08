; RUN: llc < %s

; Bug: PR31898
; XFAIL: avr

%f8 = type <8 x float>

define void @test_f8(%f8 *%P, %f8* %Q, %f8 *%S) {
  %p = load %f8, %f8* %P
  %q = load %f8, %f8* %Q
  %R = fadd %f8 %p, %q
  store %f8 %R, %f8 *%S
  ret void
}

