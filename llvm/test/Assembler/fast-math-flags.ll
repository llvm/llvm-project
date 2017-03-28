; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: opt -S < %s | FileCheck %s
; RUN: verify-uselistorder %s

@addr   = external global i64
@select = external global i1
@vec    = external global <3 x float>
@arr    = external global [3 x float]

define float @none(float %x, float %y) {
entry:
; CHECK:  %vec = load  <3 x float>,  <3 x float>* @vec
  %vec    = load  <3 x float>,  <3 x float>* @vec
; CHECK:  %select = load i1, i1* @select
  %select = load i1, i1* @select
; CHECK:  %arr    = load [3 x float], [3 x float]* @arr
  %arr    = load [3 x float], [3 x float]* @arr

; CHECK:  %a = fadd  float %x, %y
  %a = fadd  float %x, %y
; CHECK:  %a_vec = fadd  <3 x float> %vec, %vec
  %a_vec = fadd  <3 x float> %vec, %vec
; CHECK:  %b = fsub  float %x, %y
  %b = fsub  float %x, %y
; CHECK:  %b_vec = fsub  <3 x float> %vec, %vec
  %b_vec = fsub  <3 x float> %vec, %vec
; CHECK:  %c = fmul  float %x, %y
  %c = fmul  float %x, %y
; CHECK:  %c_vec = fmul  <3 x float> %vec, %vec
  %c_vec = fmul  <3 x float> %vec, %vec
; CHECK:  %d = fdiv  float %x, %y
  %d = fdiv  float %x, %y
; CHECK:  %d_vec = fdiv  <3 x float> %vec, %vec
  %d_vec = fdiv  <3 x float> %vec, %vec
; CHECK:  %e = frem  float %x, %y
  %e = frem  float %x, %y
; CHECK:  %e_vec = frem  <3 x float> %vec, %vec
  %e_vec = frem  <3 x float> %vec, %vec
; CHECK:  ret  float %e
  ret  float %e
}

; CHECK: no_nan
define float @no_nan(float %x, float %y) {
entry:
; CHECK:  %vec = load <3 x float>, <3 x float>* @vec
  %vec    = load  <3 x float>,  <3 x float>* @vec
; CHECK:  %select = load i1, i1* @select
  %select = load i1, i1* @select
; CHECK:  %arr = load  [3 x float],  [3 x float]* @arr
  %arr    = load  [3 x float],  [3 x float]* @arr

; CHECK:  %a = fadd nnan  float %x, %y
  %a = fadd nnan  float %x, %y
; CHECK:  %a_vec = fadd nnan  <3 x float> %vec, %vec
  %a_vec = fadd nnan  <3 x float> %vec, %vec
; CHECK:  %b = fsub nnan  float %x, %y
  %b = fsub nnan  float %x, %y
; CHECK:  %b_vec = fsub nnan  <3 x float> %vec, %vec
  %b_vec = fsub nnan  <3 x float> %vec, %vec
; CHECK:  %c = fmul nnan  float %x, %y
  %c = fmul nnan  float %x, %y
; CHECK:  %c_vec = fmul nnan  <3 x float> %vec, %vec
  %c_vec = fmul nnan <3 x float> %vec, %vec
; CHECK:  %d = fdiv nnan  float %x, %y
  %d = fdiv nnan float %x, %y
; CHECK:  %d_vec = fdiv nnan  <3 x float> %vec, %vec
  %d_vec = fdiv nnan <3 x float> %vec, %vec
; CHECK:  %e = frem nnan  float %x, %y
  %e = frem nnan  float %x, %y
; CHECK:  %e_vec = frem nnan  <3 x float> %vec, %vec
  %e_vec = frem nnan  <3 x float> %vec, %vec
; CHECK:  ret float %e
  ret float %e
}

; CHECK: @contract(
define float @contract(float %x, float %y) {
entry:
; CHECK: %a = fsub contract float %x, %y
  %a = fsub contract float %x, %y
; CHECK: %b = fadd contract float %x, %y
  %b = fadd contract float %x, %y
; CHECK: %c = fmul contract float %a, %b
  %c = fmul contract float %a, %b
  ret float %c
}

; CHECK: no_nan_inf
define float @no_nan_inf(float %x, float %y) {
entry:
; CHECK:  %vec = load <3 x float>, <3 x float>* @vec
  %vec    = load <3 x float>, <3 x float>* @vec
; CHECK:  %select = load i1, i1* @select
  %select = load i1, i1* @select
; CHECK:  %arr = load [3 x float], [3 x float]* @arr
  %arr    = load [3 x float], [3 x float]* @arr

; CHECK:  %a = fadd nnan ninf  float %x, %y
  %a = fadd ninf nnan  float %x, %y
; CHECK:  %a_vec = fadd nnan  <3 x float> %vec, %vec
  %a_vec = fadd nnan  <3 x float> %vec, %vec
; CHECK:  %b = fsub nnan  float %x, %y
  %b = fsub nnan  float %x, %y
; CHECK:  %b_vec = fsub nnan ninf  <3 x float> %vec, %vec
  %b_vec = fsub ninf nnan  <3 x float> %vec, %vec
; CHECK:  %c = fmul nnan  float %x, %y
  %c = fmul nnan  float %x, %y
; CHECK:  %c_vec = fmul nnan  <3 x float> %vec, %vec
  %c_vec = fmul nnan <3 x float> %vec, %vec
; CHECK:  %d = fdiv nnan ninf  float %x, %y
  %d = fdiv ninf nnan float %x, %y
; CHECK:  %d_vec = fdiv nnan  <3 x float> %vec, %vec
  %d_vec = fdiv nnan <3 x float> %vec, %vec
; CHECK:  %e = frem nnan  float %x, %y
  %e = frem nnan  float %x, %y
; CHECK:  %e_vec = frem nnan ninf  <3 x float> %vec, %vec
  %e_vec = frem ninf nnan  <3 x float> %vec, %vec
; CHECK:  ret  float %e
  ret  float %e
}

; CHECK: mixed_flags
define float @mixed_flags(float %x, float %y) {
entry:
; CHECK:  %vec = load <3 x float>, <3 x float>* @vec
  %vec    = load <3 x float>, <3 x float>* @vec
; CHECK:  %select = load i1, i1* @select
  %select = load i1, i1* @select
; CHECK:  %arr    = load [3 x float], [3 x float]* @arr
  %arr    = load [3 x float], [3 x float]* @arr

; CHECK:  %a = fadd nnan ninf float %x, %y
  %a = fadd ninf nnan float %x, %y
; CHECK:  %a_vec = fadd nnan <3 x float> %vec, %vec
  %a_vec = fadd nnan <3 x float> %vec, %vec
; CHECK:  %b = fsub fast float %x, %y
  %b = fsub nnan nsz fast float %x, %y
; CHECK:  %b_vec = fsub nnan <3 x float> %vec, %vec
  %b_vec = fsub nnan <3 x float> %vec, %vec
; CHECK:  %c = fmul fast float %x, %y
  %c = fmul nsz fast arcp float %x, %y
; CHECK:  %c_vec = fmul nsz <3 x float> %vec, %vec
  %c_vec = fmul nsz <3 x float> %vec, %vec
; CHECK:  %d = fdiv nnan ninf arcp float %x, %y
  %d = fdiv arcp ninf nnan float %x, %y
; CHECK:  %d_vec = fdiv fast <3 x float> %vec, %vec
  %d_vec = fdiv fast nnan arcp <3 x float> %vec, %vec
; CHECK:  %e = frem nnan nsz float %x, %y
  %e = frem nnan nsz float %x, %y
; CHECK:  %e_vec = frem nnan <3 x float> %vec, %vec
  %e_vec = frem nnan <3 x float> %vec, %vec
; CHECK:  ret  float %e
  ret  float %e
}
