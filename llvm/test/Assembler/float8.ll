; RUN: llvm-as < %s | llvm-dis | FileCheck %s --check-prefix=ASSEM-DISASS
; RUN: opt < %s -O3 -S | FileCheck %s --check-prefix=OPT
; RUN: verify-uselistorder %s

define float8e4m3fn @check_float8e4m3fn(float8e4m3fn %A) {
; ASSEM-DISASS: ret float8e4m3fn %A
    ret float8e4m3fn %A
}

define float8e5m2 @check_float8e5m2(float8e5m2 %A) {
; ASSEM-DISASS: ret float8e5m2 %A
    ret float8e5m2 %A
}

define float8e4m3fn @check_float8e4m3fn_literal() {
; ASSEM-DISASS: ret float8e4m3fn 0xQ31
    ret float8e4m3fn 0xQ31
}

define float8e5m2 @check_float8e5m2_literal() {
; ASSEM-DISASS: ret float8e5m2 0xS31
    ret float8e5m2 0xS31
}

define <4 x float8e4m3fn> @check_float8e4m3fn_fixed_vector() {
; ASSEM-DISASS: ret <4 x float8e4m3fn> %tmp
  %tmp = fadd <4 x float8e4m3fn> undef, undef
  ret <4 x float8e4m3fn> %tmp
}

define <4 x float8e5m2> @check_float8e5m2_fixed_vector() {
; ASSEM-DISASS: ret <4 x float8e5m2> %tmp
  %tmp = fadd <4 x float8e5m2> undef, undef
  ret <4 x float8e5m2> %tmp
}

define <vscale x 4 x float8e4m3fn> @check_float8e4m3fn_vector() {
; ASSEM-DISASS: ret <vscale x 4 x float8e4m3fn> %tmp
  %tmp = fadd <vscale x 4 x float8e4m3fn> undef, undef
  ret <vscale x 4 x float8e4m3fn> %tmp
}

define <vscale x 4 x float8e5m2> @check_float8e5m2_vector() {
; ASSEM-DISASS: ret <vscale x 4 x float8e5m2> %tmp
  %tmp = fadd <vscale x 4 x float8e5m2> undef, undef
  ret <vscale x 4 x float8e5m2> %tmp
}

define float8e4m3fn @check_float8e4m3fn_constprop() {
  %tmp = fadd float8e4m3fn 0xQ40, 0xQ40
; OPT: 0xQ48
  ret float8e4m3fn %tmp
}

define float8e5m2 @check_float8e5m2_constprop() {
  %tmp = fadd float8e5m2 0xS40, 0xS40
; OPT: 0xS44
  ret float8e5m2 %tmp
}

define float @check_float8e4m3fn_convert() {
  %tmp = fpext float8e4m3fn 0xQ40 to float
; OPT: 2.000000e+00
  ret float %tmp
}

define float @check_float8e5m2_convert() {
  %tmp = fpext float8e5m2 0xS40 to float
; OPT: 2.000000e+00
  ret float %tmp
}

; ASSEM-DISASS-LABEL @snan_float8e5m2
define float8e5m2 @snan_loat8e5m2() {
; ASSEM-DISASS: ret float8e5m2 0xS7D
    ret float8e5m2 0xS7D
}

; ASSEM-DISASS-LABEL @first_qnan_float8e5m2
define float8e5m2 @first_qnan_float8e5m2() {
; ASSEM-DISASS: ret float8e5m2 0xS7E
    ret float8e5m2 0xS7E
}

; ASSEM-DISASS-LABEL @second_qnan_float8e5m2
define float8e5m2 @second_qnan_float8e5m2() {
; ASSEM-DISASS: ret float8e5m2 0xS7F
    ret float8e5m2 0xS7F
}

; ASSEM-DISASS-LABEL @inf_float8e5m2
define float8e5m2 @inf_float8e5m2() {
; ASSEM-DISASS: ret float8e5m2 0xS7C
    ret float8e5m2 0xS7C
}

; ASSEM-DISASS-LABEL @qnan_float8e4m3fn
define float8e4m3fn @first_qnan_float8e4m3fn() {
; ASSEM-DISASS: ret float8e4m3fn 0xQ7F
    ret float8e4m3fn 0xQ7F
}