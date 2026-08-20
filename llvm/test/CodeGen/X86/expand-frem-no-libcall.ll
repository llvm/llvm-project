; This test underhandedly exploits a bug in llc's handling of -march.
; The module ends up with no triple, and thus treated as unknown arch
; with no library functions.

; RUN: llc -march=x86-64 -stop-after=expand-ir-insts %s -o - | FileCheck --check-prefix=EXPAND %s
; RUN: llc -mtriple=x86_64-linux-gnu -stop-after=expand-ir-insts %s -o - | FileCheck --check-prefix=LIBCALL %s

; When the fmod libcall is unavailable, expand-ir-insts must expand
; frem inline instead of leaving it for the DAG legalizer.

; EXPAND-LABEL: define float @frem_f32
; EXPAND-NOT: frem float
; EXPAND: fmul float

; LIBCALL-LABEL: define float @frem_f32
; LIBCALL: frem float
define float @frem_f32(float %a, float %b) {
  %r = frem float %a, %b
  ret float %r
}

; EXPAND-LABEL: define double @frem_f64
; EXPAND-NOT: frem double
; EXPAND: fmul double

; LIBCALL-LABEL: define double @frem_f64
; LIBCALL: frem double
define double @frem_f64(double %a, double %b) {
  %r = frem double %a, %b
  ret double %r
}
