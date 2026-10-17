; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -mcpu=a2 | FileCheck %s

%0 = type { double, double }

define void @maybe_an_fma(ptr sret(%0) %agg.result, ptr byval(%0) %a, ptr byval(%0) %b, ptr byval(%0) %c) nounwind {
entry:
  %a.real = load double, ptr %a
  %a.imagp = getelementptr inbounds %0, ptr %a, i32 0, i32 1
  %a.imag = load double, ptr %a.imagp
  %b.real = load double, ptr %b
  %b.imagp = getelementptr inbounds %0, ptr %b, i32 0, i32 1
  %b.imag = load double, ptr %b.imagp
  %mul.rl = fmul contract double %a.real, %b.real
  %mul.rr = fmul contract double %a.imag, %b.imag
  %mul.r = fsub contract double %mul.rl, %mul.rr
  %mul.il = fmul contract double %a.imag, %b.real
  %mul.ir = fmul contract double %a.real, %b.imag
  %mul.i = fadd contract double %mul.il, %mul.ir
  %c.real = load double, ptr %c
  %c.imagp = getelementptr inbounds %0, ptr %c, i32 0, i32 1
  %c.imag = load double, ptr %c.imagp
  %add.r = fadd contract double %mul.r, %c.real
  %add.i = fadd contract double %mul.i, %c.imag
  %imag = getelementptr inbounds %0, ptr %agg.result, i32 0, i32 1
  store double %add.r, ptr %agg.result
  store double %add.i, ptr %imag
  ret void
; CHECK: fmadd
}
