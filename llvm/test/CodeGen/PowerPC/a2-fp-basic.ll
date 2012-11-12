; RUN: llc < %s -march=ppc64 -mcpu=a2 -fp-contract=fast | FileCheck %s

%0 = type { double, double }

define void @maybe_an_fma(%0* sret %agg.result, %0* byval %a, %0* byval %b, %0* byval %c) nounwind {
entry:
  %a.realp = getelementptr inbounds %0* %a, i32 0, i32 0
  %a.real = load double* %a.realp
  %a.imagp = getelementptr inbounds %0* %a, i32 0, i32 1
  %a.imag = load double* %a.imagp
  %b.realp = getelementptr inbounds %0* %b, i32 0, i32 0
  %b.real = load double* %b.realp
  %b.imagp = getelementptr inbounds %0* %b, i32 0, i32 1
  %b.imag = load double* %b.imagp
  %mul.rl = fmul double %a.real, %b.real
  %mul.rr = fmul double %a.imag, %b.imag
  %mul.r = fsub double %mul.rl, %mul.rr
  %mul.il = fmul double %a.imag, %b.real
  %mul.ir = fmul double %a.real, %b.imag
  %mul.i = fadd double %mul.il, %mul.ir
  %c.realp = getelementptr inbounds %0* %c, i32 0, i32 0
  %c.real = load double* %c.realp
  %c.imagp = getelementptr inbounds %0* %c, i32 0, i32 1
  %c.imag = load double* %c.imagp
  %add.r = fadd double %mul.r, %c.real
  %add.i = fadd double %mul.i, %c.imag
  %real = getelementptr inbounds %0* %agg.result, i32 0, i32 0
  %imag = getelementptr inbounds %0* %agg.result, i32 0, i32 1
  store double %add.r, double* %real
  store double %add.i, double* %imag
  ret void
; CHECK: fmadd
}
