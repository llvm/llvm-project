; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu \
; RUN:   -mcpu=e500mc < %s | FileCheck %s --check-prefixes=CHECK,PPC32
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -mcpu=970 < %s | FileCheck %s --check-prefixes=CHECK,PPC64

; Without fctiwuz/fctiduz, fptoui must not convert the unadjusted source and
; select afterwards: fctiwz/fctidz set VXCVI for inputs >= 2^31/2^63, which
; traps when FP invalid exceptions are enabled. Expect a single conversion.

define i32 @fptoui_f64_i32(double %a) nounwind {
; PPC32-LABEL: fptoui_f64_i32:
; PPC32:       fctiwz
; PPC32-NOT:   fctiwz
; PPC32:       blr
  %r = fptoui double %a to i32
  ret i32 %r
}

define i64 @fptoui_f64_i64(double %a) nounwind {
; PPC64-LABEL: fptoui_f64_i64:
; PPC64:       fctidz
; PPC64-NOT:   fctidz
; PPC64:       blr
  %r = fptoui double %a to i64
  ret i64 %r
}

define i32 @fptoui_ppcf128_i32(ppc_fp128 %a) nounwind {
; CHECK-LABEL: fptoui_ppcf128_i32:
; CHECK:       fctiwz
; CHECK-NOT:   fctiwz
; CHECK:       blr
  %r = fptoui ppc_fp128 %a to i32
  ret i32 %r
}
