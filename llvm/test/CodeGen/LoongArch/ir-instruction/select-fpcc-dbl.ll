; RUN: llc --mtriple=loongarch32 --mattr=+d < %s | FileCheck %s --check-prefix=LA32
; RUN: llc --mtriple=loongarch64 --mattr=+d < %s | FileCheck %s --check-prefix=LA64

;; Test double-precision floating-point values selection after comparison

define double @fcmp_false(double %a, double %b, double %x, double %y) {
; LA32-LABEL: fcmp_false:
; LA32:       # %bb.0:
; LA32-NEXT:    fmov.d $fa0, $fa3
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fcmp_false:
; LA64:       # %bb.0:
; LA64-NEXT:    fmov.d $fa0, $fa3
; LA64-NEXT:    jirl $zero, $ra, 0
  %cmp = fcmp false double %a, %b
  %res = select i1 %cmp, double %x, double %y
  ret double %res
}

define double @fcmp_oeq(double %a, double %b, double %x, double %y) {
; LA32-LABEL: fcmp_oeq:
; LA32:       # %bb.0:
; LA32-NEXT:    fcmp.ceq.d $fcc0, $fa0, $fa1
; LA32-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fcmp_oeq:
; LA64:       # %bb.0:
; LA64-NEXT:    fcmp.ceq.d $fcc0, $fa0, $fa1
; LA64-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA64-NEXT:    jirl $zero, $ra, 0
  %cmp = fcmp oeq double %a, %b
  %res = select i1 %cmp, double %x, double %y
  ret double %res
}

define double @fcmp_ogt(double %a, double %b, double %x, double %y) {
; LA32-LABEL: fcmp_ogt:
; LA32:       # %bb.0:
; LA32-NEXT:    fcmp.clt.d $fcc0, $fa1, $fa0
; LA32-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fcmp_ogt:
; LA64:       # %bb.0:
; LA64-NEXT:    fcmp.clt.d $fcc0, $fa1, $fa0
; LA64-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA64-NEXT:    jirl $zero, $ra, 0
  %cmp = fcmp ogt double %a, %b
  %res = select i1 %cmp, double %x, double %y
  ret double %res
}

define double @fcmp_oge(double %a, double %b, double %x, double %y) {
; LA32-LABEL: fcmp_oge:
; LA32:       # %bb.0:
; LA32-NEXT:    fcmp.cle.d $fcc0, $fa1, $fa0
; LA32-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fcmp_oge:
; LA64:       # %bb.0:
; LA64-NEXT:    fcmp.cle.d $fcc0, $fa1, $fa0
; LA64-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA64-NEXT:    jirl $zero, $ra, 0
  %cmp = fcmp oge double %a, %b
  %res = select i1 %cmp, double %x, double %y
  ret double %res
}

define double @fcmp_olt(double %a, double %b, double %x, double %y) {
; LA32-LABEL: fcmp_olt:
; LA32:       # %bb.0:
; LA32-NEXT:    fcmp.clt.d $fcc0, $fa0, $fa1
; LA32-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fcmp_olt:
; LA64:       # %bb.0:
; LA64-NEXT:    fcmp.clt.d $fcc0, $fa0, $fa1
; LA64-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA64-NEXT:    jirl $zero, $ra, 0
  %cmp = fcmp olt double %a, %b
  %res = select i1 %cmp, double %x, double %y
  ret double %res
}

define double @fcmp_ole(double %a, double %b, double %x, double %y) {
; LA32-LABEL: fcmp_ole:
; LA32:       # %bb.0:
; LA32-NEXT:    fcmp.cle.d $fcc0, $fa0, $fa1
; LA32-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fcmp_ole:
; LA64:       # %bb.0:
; LA64-NEXT:    fcmp.cle.d $fcc0, $fa0, $fa1
; LA64-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA64-NEXT:    jirl $zero, $ra, 0
  %cmp = fcmp ole double %a, %b
  %res = select i1 %cmp, double %x, double %y
  ret double %res
}

define double @fcmp_one(double %a, double %b, double %x, double %y) {
; LA32-LABEL: fcmp_one:
; LA32:       # %bb.0:
; LA32-NEXT:    fcmp.cne.d $fcc0, $fa0, $fa1
; LA32-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fcmp_one:
; LA64:       # %bb.0:
; LA64-NEXT:    fcmp.cne.d $fcc0, $fa0, $fa1
; LA64-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA64-NEXT:    jirl $zero, $ra, 0
  %cmp = fcmp one double %a, %b
  %res = select i1 %cmp, double %x, double %y
  ret double %res
}

define double @fcmp_ord(double %a, double %b, double %x, double %y) {
; LA32-LABEL: fcmp_ord:
; LA32:       # %bb.0:
; LA32-NEXT:    fcmp.cor.d $fcc0, $fa0, $fa1
; LA32-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fcmp_ord:
; LA64:       # %bb.0:
; LA64-NEXT:    fcmp.cor.d $fcc0, $fa0, $fa1
; LA64-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA64-NEXT:    jirl $zero, $ra, 0
  %cmp = fcmp ord double %a, %b
  %res = select i1 %cmp, double %x, double %y
  ret double %res
}

define double @fcmp_ueq(double %a, double %b, double %x, double %y) {
; LA32-LABEL: fcmp_ueq:
; LA32:       # %bb.0:
; LA32-NEXT:    fcmp.cueq.d $fcc0, $fa0, $fa1
; LA32-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fcmp_ueq:
; LA64:       # %bb.0:
; LA64-NEXT:    fcmp.cueq.d $fcc0, $fa0, $fa1
; LA64-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA64-NEXT:    jirl $zero, $ra, 0
  %cmp = fcmp ueq double %a, %b
  %res = select i1 %cmp, double %x, double %y
  ret double %res
}

define double @fcmp_ugt(double %a, double %b, double %x, double %y) {
; LA32-LABEL: fcmp_ugt:
; LA32:       # %bb.0:
; LA32-NEXT:    fcmp.cult.d $fcc0, $fa1, $fa0
; LA32-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fcmp_ugt:
; LA64:       # %bb.0:
; LA64-NEXT:    fcmp.cult.d $fcc0, $fa1, $fa0
; LA64-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA64-NEXT:    jirl $zero, $ra, 0
  %cmp = fcmp ugt double %a, %b
  %res = select i1 %cmp, double %x, double %y
  ret double %res
}

define double @fcmp_uge(double %a, double %b, double %x, double %y) {
; LA32-LABEL: fcmp_uge:
; LA32:       # %bb.0:
; LA32-NEXT:    fcmp.cule.d $fcc0, $fa1, $fa0
; LA32-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fcmp_uge:
; LA64:       # %bb.0:
; LA64-NEXT:    fcmp.cule.d $fcc0, $fa1, $fa0
; LA64-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA64-NEXT:    jirl $zero, $ra, 0
  %cmp = fcmp uge double %a, %b
  %res = select i1 %cmp, double %x, double %y
  ret double %res
}

define double @fcmp_ult(double %a, double %b, double %x, double %y) {
; LA32-LABEL: fcmp_ult:
; LA32:       # %bb.0:
; LA32-NEXT:    fcmp.cult.d $fcc0, $fa0, $fa1
; LA32-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fcmp_ult:
; LA64:       # %bb.0:
; LA64-NEXT:    fcmp.cult.d $fcc0, $fa0, $fa1
; LA64-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA64-NEXT:    jirl $zero, $ra, 0
  %cmp = fcmp ult double %a, %b
  %res = select i1 %cmp, double %x, double %y
  ret double %res
}

define double @fcmp_ule(double %a, double %b, double %x, double %y) {
; LA32-LABEL: fcmp_ule:
; LA32:       # %bb.0:
; LA32-NEXT:    fcmp.cule.d $fcc0, $fa0, $fa1
; LA32-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fcmp_ule:
; LA64:       # %bb.0:
; LA64-NEXT:    fcmp.cule.d $fcc0, $fa0, $fa1
; LA64-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA64-NEXT:    jirl $zero, $ra, 0
  %cmp = fcmp ule double %a, %b
  %res = select i1 %cmp, double %x, double %y
  ret double %res
}

define double @fcmp_une(double %a, double %b, double %x, double %y) {
; LA32-LABEL: fcmp_une:
; LA32:       # %bb.0:
; LA32-NEXT:    fcmp.cune.d $fcc0, $fa0, $fa1
; LA32-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fcmp_une:
; LA64:       # %bb.0:
; LA64-NEXT:    fcmp.cune.d $fcc0, $fa0, $fa1
; LA64-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA64-NEXT:    jirl $zero, $ra, 0
  %cmp = fcmp une double %a, %b
  %res = select i1 %cmp, double %x, double %y
  ret double %res
}

define double @fcmp_uno(double %a, double %b, double %x, double %y) {
; LA32-LABEL: fcmp_uno:
; LA32:       # %bb.0:
; LA32-NEXT:    fcmp.cun.d $fcc0, $fa0, $fa1
; LA32-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fcmp_uno:
; LA64:       # %bb.0:
; LA64-NEXT:    fcmp.cun.d $fcc0, $fa0, $fa1
; LA64-NEXT:    fsel $fa0, $fa3, $fa2, $fcc0
; LA64-NEXT:    jirl $zero, $ra, 0
  %cmp = fcmp uno double %a, %b
  %res = select i1 %cmp, double %x, double %y
  ret double %res
}

define double @fcmp_true(double %a, double %b, double %x, double %y) {
; LA32-LABEL: fcmp_true:
; LA32:       # %bb.0:
; LA32-NEXT:    fmov.d $fa0, $fa2
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fcmp_true:
; LA64:       # %bb.0:
; LA64-NEXT:    fmov.d $fa0, $fa2
; LA64-NEXT:    jirl $zero, $ra, 0
  %cmp = fcmp true double %a, %b
  %res = select i1 %cmp, double %x, double %y
  ret double %res
}
