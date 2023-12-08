; RUN: llc -march=mips64el -mcpu=mips64r2 -O3 < %s |\
; RUN: FileCheck %s -check-prefix=HARD
; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=+soft-float < %s |\
; RUN: FileCheck %s -check-prefix=SOFT

; Check that %add is not passed in an integer register.
;
; HARD-LABEL: callfloor:
; HARD-NOT: dmfc1 $4

define double @callfloor(double %d) nounwind readnone {
entry:
  %add = fadd double %d, 1.000000e+00
  %call = tail call double @floor(double %add) nounwind readnone
  ret double %call
}

declare double @floor(double) nounwind readnone

; Check call16.
;
; SOFT-LABEL: f64add:
; SOFT: jal __adddf3

define double @f64add(double %a, double %b) {
entry:
  %add = fadd double %a, %b
  ret double %add
}
