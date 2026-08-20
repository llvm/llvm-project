; RUN: llc -O0 -fast-isel -fast-isel-abort=1 -verify-machineinstrs -mtriple=arm64-apple-darwin < %s | FileCheck %s
; RUN: llc -O0 -global-isel -verify-machineinstrs -mtriple=arm64-apple-darwin %s -o - | FileCheck %s --check-prefix=GISEL


; Materialize using fmov
define float @fmov_float1() {
; CHECK-LABEL: fmov_float1
; CHECK:       fmov s0, #1.25000000
; GISEL-LABEL: fmov_float1
; GISEL:       fmov s0, #1.25000000
  ret float 1.250000e+00
}

define float @fmov_float2() {
; CHECK-LABEL: fmov_float2
; CHECK:       fmov s0, wzr
; GISEL-LABEL: fmov_float2
; GISEL:       movi d0, #0000000000000000
  ret float 0.0e+00
}

define double @fmov_double1() {
; CHECK-LABEL: fmov_double1
; CHECK:       fmov d0, #1.25000000
; GISEL-LABEL: fmov_double1
; GISEL:       fmov d0, #1.25000000
  ret double 1.250000e+00
}

define double @fmov_double2() {
; CHECK-LABEL: fmov_double2
; CHECK:       fmov d0, xzr
; GISEL-LABEL: fmov_double2
; GISEL:       movi d0, #0000000000000000
  ret double 0.0e+00
}

; Materialize from constant pool
define float @cp_float() {
; CHECK-LABEL: cp_float
; CHECK:       adrp [[REG:x[0-9]+]], {{lCPI[0-9]+_0}}@PAGE
; CHECK-NEXT:  ldr s0, [[[REG]], {{lCPI[0-9]+_0}}@PAGEOFF]
; GISEL-LABEL: cp_float
; GISEL:       mov w8, #4059 ; =0xfdb
; GISEL-NEXT:  movk w8, #16457, lsl #16
; GISEL-NEXT:  fmov s0, w8
  ret float 0x400921FB60000000
}

define double @cp_double() {
; CHECK-LABEL: cp_double
; CHECK:       adrp [[REG:x[0-9]+]], {{lCPI[0-9]+_0}}@PAGE
; CHECK-NEXT:  ldr d0, [[[REG]], {{lCPI[0-9]+_0}}@PAGEOFF]
; GISEL-LABEL: cp_double
; GISEL:       adrp [[GREG:x[0-9]+]], {{lCPI[0-9]+_0}}@PAGE
; GISEL-NEXT:  ldr d0, [[[GREG]], {{lCPI[0-9]+_0}}@PAGEOFF]
  ret double 0x400921FB54442D18
}
