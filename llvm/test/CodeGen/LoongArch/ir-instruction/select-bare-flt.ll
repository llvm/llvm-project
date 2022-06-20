; RUN: llc --mtriple=loongarch32 --mattr=+f,-d < %s | FileCheck %s --check-prefix=LA32
; RUN: llc --mtriple=loongarch64 --mattr=+f,-d < %s | FileCheck %s --check-prefix=LA64

;; Test the bare single-precision floating-point values selection:
;; https://llvm.org/docs/LangRef.html#select-instruction

define float @test(i1 %a, float %b, float %c) {
; LA32-LABEL: test:
; LA32:       # %bb.0:
; LA32-NEXT:    andi $a0, $a0, 1
; LA32-NEXT:    movgr2cf $fcc0, $a0
; LA32-NEXT:    fsel $fa0, $fa1, $fa0, $fcc0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: test:
; LA64:       # %bb.0:
; LA64-NEXT:    andi $a0, $a0, 1
; LA64-NEXT:    movgr2cf $fcc0, $a0
; LA64-NEXT:    fsel $fa0, $fa1, $fa0, $fcc0
; LA64-NEXT:    jirl $zero, $ra, 0
  %res = select i1 %a, float %b, float %c
  ret float %res
}
