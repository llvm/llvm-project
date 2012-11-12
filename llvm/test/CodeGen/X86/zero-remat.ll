; RUN: llc < %s -march=x86-64 | FileCheck %s --check-prefix=CHECK-64
; RUN: llc < %s -march=x86-64 -o /dev/null -stats  -info-output-file - | grep asm-printer  | grep 12
; RUN: llc < %s -march=x86 | FileCheck %s --check-prefix=CHECK-32

declare void @bar(double %x)
declare void @barf(float %x)

define double @foo() nounwind {

  call void @bar(double 0.0)
  ret double 0.0

;CHECK-32: foo:
;CHECK-32: call
;CHECK-32: fldz
;CHECK-32: ret

;CHECK-64: foo:
;CHECK-64: xorps
;CHECK-64: call
;CHECK-64: xorps
;CHECK-64: ret
}


define float @foof() nounwind {
  call void @barf(float 0.0)
  ret float 0.0

;CHECK-32: foof:
;CHECK-32: call
;CHECK-32: fldz
;CHECK-32: ret

;CHECK-64: foof:
;CHECK-64: xorps
;CHECK-64: call
;CHECK-64: xorps
;CHECK-64: ret
}
