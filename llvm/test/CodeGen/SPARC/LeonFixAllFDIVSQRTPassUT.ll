; RUN: llc %s -O0 -mtriple=sparc -mcpu=leon3 -mattr=+fixallfdivsqrt -o - | FileCheck %s
; RUN: llc %s -O0 -mtriple=sparc -mcpu=ut699 -o - | FileCheck %s

; CHECK-LABEL: test_1
; CHECK:  nop
; CHECK:  nop
; CHECK:  fdivd
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
define double @test_1(ptr byval(double) %a, ptr byval(double) %b) {
entry:
    %0 = load double, ptr %a, align 8
    %1 = load double, ptr %b, align 8
    %res = fdiv double %0, %1
    ret double %res
}

declare double @llvm.sqrt.f64(double) nounwind readonly

; CHECK-LABEL: test_2
; CHECK:  nop
; CHECK:  nop
; CHECK:  nop
; CHECK:  nop
; CHECK:  nop
; CHECK:  fsqrtd
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
; CHECK-NEXT:  nop
define double @test_2(ptr byval(double) %a) {
entry:
    %0 = load double, ptr %a, align 8
    %1 = call double @llvm.sqrt.f64(double %0) nounwind
    ret double %1
}

