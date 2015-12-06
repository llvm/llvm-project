; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=nehalem -asm-verbose=false  | FileCheck %s
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=nehalem -asm-verbose=false -enable-unsafe-fp-math -enable-no-nans-fp-math  | FileCheck -check-prefix=UNSAFE %s
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=nehalem -asm-verbose=false -enable-no-nans-fp-math  | FileCheck -check-prefix=FINITE %s

; Some of these patterns can be matched as SSE min or max. Some of
; them can be matched provided that the operands are swapped.
; Some of them can't be matched at all and require a comparison
; and a conditional branch.

; The naming convention is {,x_,y_}{o,u}{gt,lt,ge,le}{,_inverse}
;  _x: use 0.0 instead of %y
;  _y: use -0.0 instead of %y
; _inverse : swap the arms of the select.

; CHECK-LABEL:      ogt:
; CHECK-NEXT: maxsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      ogt:
; UNSAFE-NEXT: maxsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ogt:
; FINITE-NEXT: maxsd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ogt(double %x, double %y) nounwind {
  %c = fcmp ogt double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK-LABEL:      olt:
; CHECK-NEXT: minsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      olt:
; UNSAFE-NEXT: minsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      olt:
; FINITE-NEXT: minsd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @olt(double %x, double %y) nounwind {
  %c = fcmp olt double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK-LABEL:      ogt_inverse:
; CHECK-NEXT: minsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      ogt_inverse:
; UNSAFE-NEXT: minsd  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ogt_inverse:
; FINITE-NEXT: minsd  %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ogt_inverse(double %x, double %y) nounwind {
  %c = fcmp ogt double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK-LABEL:      olt_inverse:
; CHECK-NEXT: maxsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      olt_inverse:
; UNSAFE-NEXT: maxsd  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      olt_inverse:
; FINITE-NEXT: maxsd  %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @olt_inverse(double %x, double %y) nounwind {
  %c = fcmp olt double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK-LABEL:      oge:
; CHECK: cmplesd %xmm0
; UNSAFE-LABEL:      oge:
; UNSAFE-NEXT: maxsd	%xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      oge:
; FINITE-NEXT: maxsd	%xmm1, %xmm0
; FINITE-NEXT: ret
define double @oge(double %x, double %y) nounwind {
  %c = fcmp oge double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK-LABEL:      ole:
; CHECK: cmplesd %xmm1
; UNSAFE-LABEL:      ole:
; UNSAFE-NEXT: minsd %xmm1, %xmm0
; FINITE-LABEL:      ole:
; FINITE-NEXT: minsd %xmm1, %xmm0
define double @ole(double %x, double %y) nounwind {
  %c = fcmp ole double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK-LABEL:      oge_inverse:
; CHECK: cmplesd %xmm0
; UNSAFE-LABEL:      oge_inverse:
; UNSAFE-NEXT: minsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      oge_inverse:
; FINITE-NEXT: minsd %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @oge_inverse(double %x, double %y) nounwind {
  %c = fcmp oge double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK-LABEL:      ole_inverse:
; CHECK: cmplesd %xmm1
; UNSAFE-LABEL:      ole_inverse:
; UNSAFE-NEXT: maxsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ole_inverse:
; FINITE-NEXT: maxsd %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ole_inverse(double %x, double %y) nounwind {
  %c = fcmp ole double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK-LABEL:      ogt_x:
; CHECK-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; CHECK-NEXT: maxsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      ogt_x:
; UNSAFE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; UNSAFE-NEXT: maxsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ogt_x:
; FINITE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; FINITE-NEXT: maxsd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ogt_x(double %x) nounwind {
  %c = fcmp ogt double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK-LABEL:      olt_x:
; CHECK-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; CHECK-NEXT: minsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      olt_x:
; UNSAFE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; UNSAFE-NEXT: minsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      olt_x:
; FINITE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; FINITE-NEXT: minsd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @olt_x(double %x) nounwind {
  %c = fcmp olt double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK-LABEL:      ogt_inverse_x:
; CHECK-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; CHECK-NEXT: minsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      ogt_inverse_x:
; UNSAFE-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; UNSAFE-NEXT: minsd  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ogt_inverse_x:
; FINITE-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; FINITE-NEXT: minsd  %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ogt_inverse_x(double %x) nounwind {
  %c = fcmp ogt double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK-LABEL:      olt_inverse_x:
; CHECK-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; CHECK-NEXT: maxsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      olt_inverse_x:
; UNSAFE-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; UNSAFE-NEXT: maxsd  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      olt_inverse_x:
; FINITE-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; FINITE-NEXT: maxsd  %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @olt_inverse_x(double %x) nounwind {
  %c = fcmp olt double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK-LABEL:      oge_x:
; CHECK:      cmplesd %xmm
; CHECK-NEXT: andpd
; UNSAFE-LABEL:      oge_x:
; UNSAFE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; UNSAFE-NEXT: maxsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      oge_x:
; FINITE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; FINITE-NEXT: maxsd   %xmm1, %xmm0
; FINITE-NEXT: ret
define double @oge_x(double %x) nounwind {
  %c = fcmp oge double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK-LABEL:      ole_x:
; CHECK:      cmplesd %xmm
; CHECK-NEXT: andpd
; UNSAFE-LABEL:      ole_x:
; UNSAFE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; UNSAFE-NEXT: minsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ole_x:
; FINITE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; FINITE-NEXT: minsd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ole_x(double %x) nounwind {
  %c = fcmp ole double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK-LABEL:      oge_inverse_x:
; CHECK:      cmplesd %xmm
; CHECK-NEXT: andnpd
; UNSAFE-LABEL:      oge_inverse_x:
; UNSAFE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; UNSAFE-NEXT: minsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      oge_inverse_x:
; FINITE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; FINITE-NEXT: minsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @oge_inverse_x(double %x) nounwind {
  %c = fcmp oge double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK-LABEL:      ole_inverse_x:
; CHECK:      cmplesd %xmm
; UNSAFE-LABEL:      ole_inverse_x:
; UNSAFE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; UNSAFE-NEXT: maxsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ole_inverse_x:
; FINITE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; FINITE-NEXT: maxsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ole_inverse_x(double %x) nounwind {
  %c = fcmp ole double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK-LABEL:      ugt:
; CHECK:      cmpnlesd %xmm1
; UNSAFE-LABEL:      ugt:
; UNSAFE-NEXT: maxsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ugt:
; FINITE-NEXT: maxsd   %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ugt(double %x, double %y) nounwind {
  %c = fcmp ugt double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK-LABEL:      ult:
; CHECK:      cmpnlesd %xmm0
; UNSAFE-LABEL:      ult:
; UNSAFE-NEXT: minsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ult:
; FINITE-NEXT: minsd   %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ult(double %x, double %y) nounwind {
  %c = fcmp ult double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK-LABEL:      ugt_inverse:
; CHECK:      cmpnlesd %xmm1
; UNSAFE-LABEL:      ugt_inverse:
; UNSAFE-NEXT: minsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ugt_inverse:
; FINITE-NEXT: minsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ugt_inverse(double %x, double %y) nounwind {
  %c = fcmp ugt double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK-LABEL:      ult_inverse:
; CHECK:      cmpnlesd %xmm0
; UNSAFE-LABEL:      ult_inverse:
; UNSAFE-NEXT: maxsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ult_inverse:
; FINITE-NEXT: maxsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ult_inverse(double %x, double %y) nounwind {
  %c = fcmp ult double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK-LABEL:      uge:
; CHECK-NEXT: maxsd   %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      uge:
; UNSAFE-NEXT: maxsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      uge:
; FINITE-NEXT: maxsd   %xmm1, %xmm0
; FINITE-NEXT: ret
define double @uge(double %x, double %y) nounwind {
  %c = fcmp uge double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK-LABEL:      ule:
; CHECK-NEXT: minsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      ule:
; UNSAFE-NEXT: minsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ule:
; FINITE-NEXT: minsd   %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ule(double %x, double %y) nounwind {
  %c = fcmp ule double %x, %y
  %d = select i1 %c, double %x, double %y
  ret double %d
}

; CHECK-LABEL:      uge_inverse:
; CHECK-NEXT: minsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      uge_inverse:
; UNSAFE-NEXT: minsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      uge_inverse:
; FINITE-NEXT: minsd %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @uge_inverse(double %x, double %y) nounwind {
  %c = fcmp uge double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK-LABEL:      ule_inverse:
; CHECK-NEXT: maxsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      ule_inverse:
; UNSAFE-NEXT: maxsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ule_inverse:
; FINITE-NEXT: maxsd %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ule_inverse(double %x, double %y) nounwind {
  %c = fcmp ule double %x, %y
  %d = select i1 %c, double %y, double %x
  ret double %d
}

; CHECK-LABEL:      ugt_x:
; CHECK:      cmpnlesd %xmm
; CHECK-NEXT: andpd
; UNSAFE-LABEL:      ugt_x:
; UNSAFE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; UNSAFE-NEXT: maxsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ugt_x:
; FINITE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; FINITE-NEXT: maxsd   %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ugt_x(double %x) nounwind {
  %c = fcmp ugt double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK-LABEL:      ult_x:
; CHECK:      cmpnlesd %xmm
; CHECK-NEXT: andpd
; UNSAFE-LABEL:      ult_x:
; UNSAFE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; UNSAFE-NEXT: minsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ult_x:
; FINITE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; FINITE-NEXT: minsd   %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ult_x(double %x) nounwind {
  %c = fcmp ult double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK-LABEL:      ugt_inverse_x:
; CHECK:      cmpnlesd %xmm
; CHECK-NEXT: andnpd
; UNSAFE-LABEL:      ugt_inverse_x:
; UNSAFE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; UNSAFE-NEXT: minsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ugt_inverse_x:
; FINITE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; FINITE-NEXT: minsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ugt_inverse_x(double %x) nounwind {
  %c = fcmp ugt double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK-LABEL:      ult_inverse_x:
; CHECK:      cmpnlesd %xmm
; CHECK-NEXT: andnpd
; UNSAFE-LABEL:      ult_inverse_x:
; UNSAFE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; UNSAFE-NEXT: maxsd   %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ult_inverse_x:
; FINITE-NEXT: xorp{{[sd]}}   %xmm1, %xmm1
; FINITE-NEXT: maxsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ult_inverse_x(double %x) nounwind {
  %c = fcmp ult double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK-LABEL:      uge_x:
; CHECK-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; CHECK-NEXT: maxsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      uge_x:
; UNSAFE-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; UNSAFE-NEXT: maxsd  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      uge_x:
; FINITE-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; FINITE-NEXT: maxsd  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @uge_x(double %x) nounwind {
  %c = fcmp uge double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK-LABEL:      ule_x:
; CHECK-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; CHECK-NEXT: minsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      ule_x:
; UNSAFE-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; UNSAFE-NEXT: minsd  %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ule_x:
; FINITE-NEXT: xorp{{[sd]}}  %xmm1, %xmm1
; FINITE-NEXT: minsd  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ule_x(double %x) nounwind {
  %c = fcmp ule double %x, 0.000000e+00
  %d = select i1 %c, double %x, double 0.000000e+00
  ret double %d
}

; CHECK-LABEL:      uge_inverse_x:
; CHECK-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; CHECK-NEXT: minsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      uge_inverse_x:
; UNSAFE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; UNSAFE-NEXT: minsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      uge_inverse_x:
; FINITE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; FINITE-NEXT: minsd %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @uge_inverse_x(double %x) nounwind {
  %c = fcmp uge double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK-LABEL:      ule_inverse_x:
; CHECK-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; CHECK-NEXT: maxsd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      ule_inverse_x:
; UNSAFE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; UNSAFE-NEXT: maxsd %xmm1, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ule_inverse_x:
; FINITE-NEXT: xorp{{[sd]}} %xmm1, %xmm1
; FINITE-NEXT: maxsd %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ule_inverse_x(double %x) nounwind {
  %c = fcmp ule double %x, 0.000000e+00
  %d = select i1 %c, double 0.000000e+00, double %x
  ret double %d
}

; CHECK-LABEL:      ogt_y:
; CHECK-NEXT: maxsd {{[^,]*}}, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      ogt_y:
; UNSAFE-NEXT: maxsd {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ogt_y:
; FINITE-NEXT: maxsd {{[^,]*}}, %xmm0
; FINITE-NEXT: ret
define double @ogt_y(double %x) nounwind {
  %c = fcmp ogt double %x, -0.000000e+00
  %d = select i1 %c, double %x, double -0.000000e+00
  ret double %d
}

; CHECK-LABEL:      olt_y:
; CHECK-NEXT: minsd {{[^,]*}}, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      olt_y:
; UNSAFE-NEXT: minsd {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      olt_y:
; FINITE-NEXT: minsd {{[^,]*}}, %xmm0
; FINITE-NEXT: ret
define double @olt_y(double %x) nounwind {
  %c = fcmp olt double %x, -0.000000e+00
  %d = select i1 %c, double %x, double -0.000000e+00
  ret double %d
}

; CHECK-LABEL:      ogt_inverse_y:
; CHECK-NEXT: movsd  {{[^,]*}}, %xmm1
; CHECK-NEXT: minsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      ogt_inverse_y:
; UNSAFE-NEXT: minsd  {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ogt_inverse_y:
; FINITE-NEXT: movsd  {{[^,]*}}, %xmm1
; FINITE-NEXT: minsd  %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ogt_inverse_y(double %x) nounwind {
  %c = fcmp ogt double %x, -0.000000e+00
  %d = select i1 %c, double -0.000000e+00, double %x
  ret double %d
}

; CHECK-LABEL:      olt_inverse_y:
; CHECK-NEXT: movsd  {{[^,]*}}, %xmm1
; CHECK-NEXT: maxsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      olt_inverse_y:
; UNSAFE-NEXT: maxsd  {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      olt_inverse_y:
; FINITE-NEXT: movsd  {{[^,]*}}, %xmm1
; FINITE-NEXT: maxsd  %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @olt_inverse_y(double %x) nounwind {
  %c = fcmp olt double %x, -0.000000e+00
  %d = select i1 %c, double -0.000000e+00, double %x
  ret double %d
}

; CHECK-LABEL:      oge_y:
; CHECK:      cmplesd %xmm0
; UNSAFE-LABEL:      oge_y:
; UNSAFE-NEXT: maxsd   {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      oge_y:
; FINITE-NEXT: maxsd   {{[^,]*}}, %xmm0
; FINITE-NEXT: ret
define double @oge_y(double %x) nounwind {
  %c = fcmp oge double %x, -0.000000e+00
  %d = select i1 %c, double %x, double -0.000000e+00
  ret double %d
}

; CHECK-LABEL:      ole_y:
; CHECK:      cmplesd %xmm
; UNSAFE-LABEL:      ole_y:
; UNSAFE-NEXT: minsd {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ole_y:
; FINITE-NEXT: minsd {{[^,]*}}, %xmm0
; FINITE-NEXT: ret
define double @ole_y(double %x) nounwind {
  %c = fcmp ole double %x, -0.000000e+00
  %d = select i1 %c, double %x, double -0.000000e+00
  ret double %d
}

; CHECK-LABEL:      oge_inverse_y:
; CHECK:      cmplesd %xmm0
; UNSAFE-LABEL:      oge_inverse_y:
; UNSAFE-NEXT: minsd   {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      oge_inverse_y:
; FINITE-NEXT: movsd   {{[^,]*}}, %xmm1
; FINITE-NEXT: minsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @oge_inverse_y(double %x) nounwind {
  %c = fcmp oge double %x, -0.000000e+00
  %d = select i1 %c, double -0.000000e+00, double %x
  ret double %d
}

; CHECK-LABEL:      ole_inverse_y:
; CHECK:      cmplesd %xmm
; UNSAFE-LABEL:      ole_inverse_y:
; UNSAFE-NEXT: maxsd   {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ole_inverse_y:
; FINITE-NEXT: movsd   {{[^,]*}}, %xmm1
; FINITE-NEXT: maxsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ole_inverse_y(double %x) nounwind {
  %c = fcmp ole double %x, -0.000000e+00
  %d = select i1 %c, double -0.000000e+00, double %x
  ret double %d
}

; CHECK-LABEL:      ugt_y:
; CHECK:      cmpnlesd %xmm
; UNSAFE-LABEL:      ugt_y:
; UNSAFE-NEXT: maxsd   {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ugt_y:
; FINITE-NEXT: maxsd   {{[^,]*}}, %xmm0
; FINITE-NEXT: ret
define double @ugt_y(double %x) nounwind {
  %c = fcmp ugt double %x, -0.000000e+00
  %d = select i1 %c, double %x, double -0.000000e+00
  ret double %d
}

; CHECK-LABEL:      ult_y:
; CHECK:      cmpnlesd %xmm0
; UNSAFE-LABEL:      ult_y:
; UNSAFE-NEXT: minsd   {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ult_y:
; FINITE-NEXT: minsd   {{[^,]*}}, %xmm0
; FINITE-NEXT: ret
define double @ult_y(double %x) nounwind {
  %c = fcmp ult double %x, -0.000000e+00
  %d = select i1 %c, double %x, double -0.000000e+00
  ret double %d
}

; CHECK-LABEL:      ugt_inverse_y:
; CHECK:      cmpnlesd %xmm
; UNSAFE-LABEL:      ugt_inverse_y:
; UNSAFE-NEXT: minsd   {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ugt_inverse_y:
; FINITE-NEXT: movsd   {{[^,]*}}, %xmm1
; FINITE-NEXT: minsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ugt_inverse_y(double %x) nounwind {
  %c = fcmp ugt double %x, -0.000000e+00
  %d = select i1 %c, double -0.000000e+00, double %x
  ret double %d
}

; CHECK-LABEL:      ult_inverse_y:
; CHECK:      cmpnlesd %xmm
; UNSAFE-LABEL:      ult_inverse_y:
; UNSAFE-NEXT: maxsd   {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ult_inverse_y:
; FINITE-NEXT: movsd   {{[^,]*}}, %xmm1
; FINITE-NEXT: maxsd   %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}}  %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ult_inverse_y(double %x) nounwind {
  %c = fcmp ult double %x, -0.000000e+00
  %d = select i1 %c, double -0.000000e+00, double %x
  ret double %d
}

; CHECK-LABEL:      uge_y:
; CHECK-NEXT: movsd  {{[^,]*}}, %xmm1
; CHECK-NEXT: maxsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      uge_y:
; UNSAFE-NEXT: maxsd  {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      uge_y:
; FINITE-NEXT: maxsd  {{[^,]*}}, %xmm0
; FINITE-NEXT: ret
define double @uge_y(double %x) nounwind {
  %c = fcmp uge double %x, -0.000000e+00
  %d = select i1 %c, double %x, double -0.000000e+00
  ret double %d
}

; CHECK-LABEL:      ule_y:
; CHECK-NEXT: movsd  {{[^,]*}}, %xmm1
; CHECK-NEXT: minsd  %xmm0, %xmm1
; CHECK-NEXT: movap{{[sd]}} %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      ule_y:
; UNSAFE-NEXT: minsd  {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ule_y:
; FINITE-NEXT: minsd  {{[^,]*}}, %xmm0
; FINITE-NEXT: ret
define double @ule_y(double %x) nounwind {
  %c = fcmp ule double %x, -0.000000e+00
  %d = select i1 %c, double %x, double -0.000000e+00
  ret double %d
}

; CHECK-LABEL:      uge_inverse_y:
; CHECK-NEXT: minsd {{[^,]*}}, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      uge_inverse_y:
; UNSAFE-NEXT: minsd {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      uge_inverse_y:
; FINITE-NEXT: movsd {{[^,]*}}, %xmm1
; FINITE-NEXT: minsd %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @uge_inverse_y(double %x) nounwind {
  %c = fcmp uge double %x, -0.000000e+00
  %d = select i1 %c, double -0.000000e+00, double %x
  ret double %d
}

; CHECK-LABEL:      ule_inverse_y:
; CHECK-NEXT: maxsd {{[^,]*}}, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL:      ule_inverse_y:
; UNSAFE-NEXT: maxsd {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL:      ule_inverse_y:
; FINITE-NEXT: movsd {{[^,]*}}, %xmm1
; FINITE-NEXT: maxsd %xmm0, %xmm1
; FINITE-NEXT: movap{{[sd]}} %xmm1, %xmm0
; FINITE-NEXT: ret
define double @ule_inverse_y(double %x) nounwind {
  %c = fcmp ule double %x, -0.000000e+00
  %d = select i1 %c, double -0.000000e+00, double %x
  ret double %d
}
; Test a few more misc. cases.

; CHECK-LABEL: clampTo3k_a:
; CHECK-NEXT: movsd {{[^,]*}}, %xmm1
; CHECK-NEXT: minsd %xmm0, %xmm1
; CHECK-NEXT: movapd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL: clampTo3k_a:
; UNSAFE-NEXT: minsd {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL: clampTo3k_a:
; FINITE-NEXT: movsd {{[^,]*}}, %xmm1
; FINITE-NEXT: minsd %xmm0, %xmm1
; FINITE-NEXT: movapd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @clampTo3k_a(double %x) nounwind readnone {
entry:
  %0 = fcmp ogt double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK-LABEL: clampTo3k_b:
; CHECK-NEXT: minsd {{[^,]*}}, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL: clampTo3k_b:
; UNSAFE-NEXT: minsd {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL: clampTo3k_b:
; FINITE-NEXT: movsd {{[^,]*}}, %xmm1
; FINITE-NEXT: minsd %xmm0, %xmm1
; FINITE-NEXT: movapd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @clampTo3k_b(double %x) nounwind readnone {
entry:
  %0 = fcmp uge double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK-LABEL: clampTo3k_c:
; CHECK-NEXT: movsd {{[^,]*}}, %xmm1
; CHECK-NEXT: maxsd %xmm0, %xmm1
; CHECK-NEXT: movapd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL: clampTo3k_c:
; UNSAFE-NEXT: maxsd {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL: clampTo3k_c:
; FINITE-NEXT: movsd {{[^,]*}}, %xmm1
; FINITE-NEXT: maxsd %xmm0, %xmm1
; FINITE-NEXT: movapd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @clampTo3k_c(double %x) nounwind readnone {
entry:
  %0 = fcmp olt double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK-LABEL: clampTo3k_d:
; CHECK-NEXT: maxsd {{[^,]*}}, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL: clampTo3k_d:
; UNSAFE-NEXT: maxsd {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL: clampTo3k_d:
; FINITE-NEXT: movsd {{[^,]*}}, %xmm1
; FINITE-NEXT: maxsd %xmm0, %xmm1
; FINITE-NEXT: movapd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @clampTo3k_d(double %x) nounwind readnone {
entry:
  %0 = fcmp ule double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK-LABEL: clampTo3k_e:
; CHECK-NEXT: movsd {{[^,]*}}, %xmm1
; CHECK-NEXT: maxsd %xmm0, %xmm1
; CHECK-NEXT: movapd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL: clampTo3k_e:
; UNSAFE-NEXT: maxsd {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL: clampTo3k_e:
; FINITE-NEXT: movsd {{[^,]*}}, %xmm1
; FINITE-NEXT: maxsd %xmm0, %xmm1
; FINITE-NEXT: movapd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @clampTo3k_e(double %x) nounwind readnone {
entry:
  %0 = fcmp olt double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK-LABEL: clampTo3k_f:
; CHECK-NEXT: maxsd {{[^,]*}}, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL: clampTo3k_f:
; UNSAFE-NEXT: maxsd {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL: clampTo3k_f:
; FINITE-NEXT: movsd {{[^,]*}}, %xmm1
; FINITE-NEXT: maxsd %xmm0, %xmm1
; FINITE-NEXT: movapd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @clampTo3k_f(double %x) nounwind readnone {
entry:
  %0 = fcmp ule double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK-LABEL: clampTo3k_g:
; CHECK-NEXT: movsd {{[^,]*}}, %xmm1
; CHECK-NEXT: minsd %xmm0, %xmm1
; CHECK-NEXT: movapd %xmm1, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL: clampTo3k_g:
; UNSAFE-NEXT: minsd {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL: clampTo3k_g:
; FINITE-NEXT: movsd {{[^,]*}}, %xmm1
; FINITE-NEXT: minsd %xmm0, %xmm1
; FINITE-NEXT: movapd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @clampTo3k_g(double %x) nounwind readnone {
entry:
  %0 = fcmp ogt double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; CHECK-LABEL: clampTo3k_h:
; CHECK-NEXT: minsd {{[^,]*}}, %xmm0
; CHECK-NEXT: ret
; UNSAFE-LABEL: clampTo3k_h:
; UNSAFE-NEXT: minsd {{[^,]*}}, %xmm0
; UNSAFE-NEXT: ret
; FINITE-LABEL: clampTo3k_h:
; FINITE-NEXT: movsd {{[^,]*}}, %xmm1
; FINITE-NEXT: minsd %xmm0, %xmm1
; FINITE-NEXT: movapd %xmm1, %xmm0
; FINITE-NEXT: ret
define double @clampTo3k_h(double %x) nounwind readnone {
entry:
  %0 = fcmp uge double %x, 3.000000e+03           ; <i1> [#uses=1]
  %x_addr.0 = select i1 %0, double 3.000000e+03, double %x ; <double> [#uses=1]
  ret double %x_addr.0
}

; UNSAFE-LABEL: test_maxpd:
; UNSAFE-NEXT: maxpd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define <2 x double> @test_maxpd(<2 x double> %x, <2 x double> %y) nounwind {
  %max_is_x = fcmp oge <2 x double> %x, %y
  %max = select <2 x i1> %max_is_x, <2 x double> %x, <2 x double> %y
  ret <2 x double> %max
}

; UNSAFE-LABEL: test_minpd:
; UNSAFE-NEXT: minpd %xmm1, %xmm0
; UNSAFE-NEXT: ret
define <2 x double> @test_minpd(<2 x double> %x, <2 x double> %y) nounwind {
  %min_is_x = fcmp ole <2 x double> %x, %y
  %min = select <2 x i1> %min_is_x, <2 x double> %x, <2 x double> %y
  ret <2 x double> %min
}

; UNSAFE-LABEL: test_maxps:
; UNSAFE-NEXT: maxps %xmm1, %xmm0
; UNSAFE-NEXT: ret
define <4 x float> @test_maxps(<4 x float> %x, <4 x float> %y) nounwind {
  %max_is_x = fcmp oge <4 x float> %x, %y
  %max = select <4 x i1> %max_is_x, <4 x float> %x, <4 x float> %y
  ret <4 x float> %max
}

; UNSAFE-LABEL: test_minps:
; UNSAFE-NEXT: minps %xmm1, %xmm0
; UNSAFE-NEXT: ret
define <4 x float> @test_minps(<4 x float> %x, <4 x float> %y) nounwind {
  %min_is_x = fcmp ole <4 x float> %x, %y
  %min = select <4 x i1> %min_is_x, <4 x float> %x, <4 x float> %y
  ret <4 x float> %min
}

; UNSAFE-LABEL: test_maxps_illegal_v2f32:
; UNSAFE-NEXT: maxps %xmm1, %xmm0
; UNSAFE-NEXT: ret
define <2 x float> @test_maxps_illegal_v2f32(<2 x float> %x, <2 x float> %y) nounwind {
  %max_is_x = fcmp oge <2 x float> %x, %y
  %max = select <2 x i1> %max_is_x, <2 x float> %x, <2 x float> %y
  ret <2 x float> %max
}

; UNSAFE-LABEL: test_minps_illegal_v2f32:
; UNSAFE-NEXT: minps %xmm1, %xmm0
; UNSAFE-NEXT: ret
define <2 x float> @test_minps_illegal_v2f32(<2 x float> %x, <2 x float> %y) nounwind {
  %min_is_x = fcmp ole <2 x float> %x, %y
  %min = select <2 x i1> %min_is_x, <2 x float> %x, <2 x float> %y
  ret <2 x float> %min
}

; UNSAFE-LABEL: test_maxps_illegal_v3f32:
; UNSAFE-NEXT: maxps %xmm1, %xmm0
; UNSAFE-NEXT: ret
define <3 x float> @test_maxps_illegal_v3f32(<3 x float> %x, <3 x float> %y) nounwind {
  %max_is_x = fcmp oge <3 x float> %x, %y
  %max = select <3 x i1> %max_is_x, <3 x float> %x, <3 x float> %y
  ret <3 x float> %max
}

; UNSAFE-LABEL: test_minps_illegal_v3f32:
; UNSAFE-NEXT: minps %xmm1, %xmm0
; UNSAFE-NEXT: ret
define <3 x float> @test_minps_illegal_v3f32(<3 x float> %x, <3 x float> %y) nounwind {
  %min_is_x = fcmp ole <3 x float> %x, %y
  %min = select <3 x i1> %min_is_x, <3 x float> %x, <3 x float> %y
  ret <3 x float> %min
}
