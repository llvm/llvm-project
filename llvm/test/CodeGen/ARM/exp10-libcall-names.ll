; RUN: llc -mtriple=armv7-linux-gnu < %s | FileCheck -check-prefix=LINUX %s
; RUN: llc -mtriple=armv7-apple-macos10.9 < %s | FileCheck -check-prefix=APPLE %s
; RUN: llc -mtriple=armv7-apple-ios7.0 < %s | FileCheck -check-prefix=APPLE %s
; RUN: llc -mtriple=armv7-apple-tvos7.0 < %s | FileCheck -check-prefix=APPLE %s
; RUN: llc -mtriple=armv7-apple-watchos7.0 < %s | FileCheck -check-prefix=APPLE %s
; RUN: llc -mtriple=armv7-apple-xros7.0 < %s | FileCheck -check-prefix=APPLE %s

; RUN: not llc -mtriple=armv7-apple-macos10.8 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -mtriple=armv7-apple-ios6.0 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -mtriple=armv7-apple-tvos6.0 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -mtriple=armv7-apple-xros6.0 -filetype=null %s 2>&1 | FileCheck -check-prefix=ERR %s

; Check exp10/exp10f is emitted as __exp10/__exp10f on assorted systems.

; ERR: no libcall available for fexp10

define float @test_exp10_f32(float %x) {
; LINUX-LABEL: test_exp10_f32:
; LINUX:       @ %bb.0:
; LINUX-NEXT:    b exp10f
;
; APPLE-LABEL: test_exp10_f32:
; APPLE:       @ %bb.0:
; APPLE-NEXT:    b ___exp10f
  %ret = call float @llvm.exp10.f32(float %x)
  ret float %ret
}

define double @test_exp10_f64(double %x) {
; LINUX-LABEL: test_exp10_f64:
; LINUX:       @ %bb.0:
; LINUX-NEXT:    b exp10
;
; APPLE-LABEL: test_exp10_f64:
; APPLE:       @ %bb.0:
; APPLE-NEXT:    b ___exp10
  %ret = call double @llvm.exp10.f64(double %x)
  ret double %ret
}
