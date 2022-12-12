; RUN: llc < %s -mtriple=i686-unknown-unknown | FileCheck %s --check-prefixes=CHECK,X86
; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s --check-prefixes=CHECK,X64

define i129 @fptosi_float(float %a) nounwind {
; CHECK-LABEL: fptosi_float:
; CHECK-NOT:     call
  %res = fptosi float %a to i129
  ret i129 %res
}

define i129 @fptosi_double(double %a) nounwind {
; CHECK-LABEL: fptosi_double:
; CHECK-NOT:     call
  %res = fptosi double %a to i129
  ret i129 %res
}

define i129 @fptosi_fp128(fp128 %a) nounwind {
; CHECK-LABEL: fptosi_fp128:
; CHECK-NOT:     call
  %res = fptosi fp128 %a to i129
  ret i129 %res
}

define i129 @fptoui_float(float %a) nounwind {
; CHECK-LABEL: fptoui_float:
; CHECK-NOT:     call
  %res = fptoui float %a to i129
  ret i129 %res
}

define i129 @fptoui_double(double %a) nounwind {
; CHECK-LABEL: fptoui_double:
; CHECK-NOT:     call
  %res = fptoui double %a to i129
  ret i129 %res
}

define i129 @fptoui_fp128(fp128 %a) nounwind {
; CHECK-LABEL: fptoui_fp128:
; CHECK-NOT:     call
  %res = fptoui fp128 %a to i129
  ret i129 %res
}

define float @sitofp_float(i129 %a) nounwind {
; CHECK-LABEL: sitofp_float:
; CHECK-NOT:     call
  %res = sitofp i129 %a to float
  ret float %res
}

define double @sitofp_double(i129 %a) nounwind {
; CHECK-LABEL: sitofp_double:
; CHECK-NOT:     call
  %res = sitofp i129 %a to double
  ret double %res
}

define fp128 @sitofp_fp128(i129 %a) nounwind {
; CHECK-LABEL: sitofp_fp128:
; CHECK-NOT:     call
  %res = sitofp i129 %a to fp128
  ret fp128 %res
}

define float @uitofp_float(i129 %a) nounwind {
; CHECK-LABEL: uitofp_float:
; CHECK-NOT:     call
  %res = uitofp i129 %a to float
  ret float %res
}

define double @uitofp_double(i129 %a) nounwind {
; CHECK-LABEL: uitofp_double:
; CHECK-NOT:     call
  %res = uitofp i129 %a to double
  ret double %res
}

define fp128 @uitofp_fp128(i129 %a) nounwind {
; CHECK-LABEL: uitofp_fp128:
; CHECK-NOT:     call
  %res = uitofp i129 %a to fp128
  ret fp128 %res
}

; higher sizes
define i257 @fptosi257_double(double %a) nounwind {
; CHECK-LABEL: fptosi257_double:
; CHECK-NOT:     call
  %res = fptosi double %a to i257
  ret i257 %res
}

; half tests
define i257 @fptosi_half(half %a) nounwind {
; X86-LABEL: fptosi_half:
; X86: __gnu_h2f_ieee
;
; X64-LABEL: fptosi_half:
; X64: uitofp_half
  %res = fptosi half %a to i257
  ret i257 %res
}

define half @uitofp_half(i257 %a) nounwind {
; X86-LABEL: uitofp_half:
; X86: __gnu_f2h_ieee
;
; X64-LABEL: uitofp_half:
; X64: uitofp_half
  %res = uitofp i257 %a to half
  ret half %res
}

; x86_fp80 tests
define i257 @fptoui_x86_fp80(x86_fp80 %a) nounwind {
; CHECK-LABEL: fptoui_x86_fp80:
; CHECK: __extendxftf2
  %res = fptoui x86_fp80 %a to i257
  ret i257 %res
}

define x86_fp80 @sitofp_x86_fp80(i257 %a) nounwind {
; CHECK-LABEL: sitofp_x86_fp80:
; CHECK: __trunctfxf2
  %res = sitofp i257 %a to x86_fp80
  ret x86_fp80 %res
}
