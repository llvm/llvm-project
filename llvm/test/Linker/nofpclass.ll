; RUN: llvm-link %s %S/Inputs/nofpclass.ll -S -o - | FileCheck -check-prefix=ORDER1 %s
; RUN: llvm-link %S/Inputs/nofpclass.ll %s -S -o - | FileCheck -check-prefix=ORDER2 %s

; Make sure nofpclass is dropped if the function was declared as
; nofpclass, but not defined with nofpclass.

; ORDER1: define float @caller(float %arg) {
; ORDER1-NEXT: %result = call float @declared_as_nonan(float %arg)
; ORDER1-NEXT: ret float %result

; ORDER1: define float @declared_as_nonan(float %arg) {
; ORDER1-NEXT: %add = fadd float %arg, 1.000000e+00
; ORDER1-NEXT: ret float %add


; ORDER2: define float @declared_as_nonan(float %arg) {
; ORDER2-NEXT: %add = fadd float %arg, 1.000000e+00
; ORDER2-NEXT: ret float %add

; ORDER2: define float @caller(float %arg) {
; ORDER2-NEXT: %result = call float @declared_as_nonan(float %arg)
; ORDER2-NEXT: ret float %result


declare nofpclass(nan) float @declared_as_nonan(float nofpclass(nan))

define float @caller(float %arg) {
  %result = call float @declared_as_nonan(float %arg)
  ret float %result
}
