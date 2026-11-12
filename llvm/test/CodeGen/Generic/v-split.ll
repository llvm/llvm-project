; RUN: llc < %s

%f8 = type <8 x float>

define void @test_f8(ptr %P, ptr %Q, ptr %S) {
  %p = load %f8, ptr %P
  %q = load %f8, ptr %Q
  %R = fadd %f8 %p, %q
  store %f8 %R, ptr %S
  ret void
}

