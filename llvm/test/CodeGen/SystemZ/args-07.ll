; Test multiple return values (LLVM ABI extension)
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -verify-machineinstrs| FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -verify-machineinstrs \
; RUN:   -mattr=soft-float | FileCheck %s --check-prefix=SOFT-FLOAT

; Up to four integer return values fit into GPRs.
define { i64, i64, i64, i64 } @f1() {
; CHECK-LABEL: f1:
; CHECK: lghi %r2, 0
; CHECK: lghi %r3, 1
; CHECK: lghi %r4, 2
; CHECK: lghi %r5, 3
; CHECK: br %r14
  ret { i64, i64, i64, i64 } { i64 0, i64 1, i64 2, i64 3 }
}

; More than four integer return values use sret.
define { i64, i64, i64, i64, i64 } @f2() {
; CHECK-LABEL: f2:
; CHECK: mvghi 32(%r2), 4
; CHECK: mvghi 24(%r2), 3
; CHECK: mvghi 16(%r2), 2
; CHECK: mvghi 8(%r2), 1
; CHECK: mvghi 0(%r2), 0
; CHECK: br %r14
  ret { i64, i64, i64, i64, i64 } { i64 0, i64 1, i64 2, i64 3, i64 4 }
}

; Up to four floating-point return values fit into FPRs.
define { double, double, double, double } @f3() {
; CHECK-LABEL: f3:
; CHECK: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: ld %f0, 0([[TMP]])
; CHECK: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: ld %f2, 0([[TMP]])
; CHECK: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: ld %f4, 0([[TMP]])
; CHECK: larl [[TMP:%r[0-5]]], .LCPI
; CHECK: ld %f6, 0([[TMP]])
; CHECK: br %r14

; SOFT-FLOAT-LABEL: f3:
; SOFT-FLOAT-NOT: %{{[fv]}}
; SOFT-FLOAT: llihh   %r2, 16368
; SOFT-FLOAT-NEXT: llihh   %r3, 16384
; SOFT-FLOAT-NEXT: llihh   %r4, 16392
; SOFT-FLOAT-NEXT: llihh   %r5, 16400
; SOFT-FLOAT-NEXT: br      %r14
  ret { double, double, double, double }
      { double 1.0, double 2.0, double 3.0, double 4.0 }
}

; More than four floating-point return values use sret.
define { double, double, double, double, double } @f4() {
; CHECK-LABEL: f4:
; CHECK: llihh [[TMP:%r[0-5]]], 16404
; CHECK: stg [[TMP]], 32(%r2)
; CHECK: llihh [[TMP:%r[0-5]]], 16400
; CHECK: stg [[TMP]], 24(%r2)
; CHECK: llihh [[TMP:%r[0-5]]], 16392
; CHECK: stg [[TMP]], 16(%r2)
; CHECK: llihh [[TMP:%r[0-5]]], 16384
; CHECK: stg [[TMP]], 8(%r2)
; CHECK: llihh [[TMP:%r[0-5]]], 16368
; CHECK: stg [[TMP]], 0(%r2)
; CHECK: br %r14

; SOFT-FLOAT-LABEL: f4:
; SOFT-FLOAT-NOT: %{{[fv]}}
; SOFT-FLOAT-NOT: %r2
; SOFT-FLOAT: llihh   %r0, 16404
; SOFT-FLOAT-NEXT: stg     %r0, 32(%r2)
; SOFT-FLOAT-NEXT: llihh   %r0, 16400
; SOFT-FLOAT-NEXT: stg     %r0, 24(%r2)
; SOFT-FLOAT-NEXT: llihh   %r0, 16392
; SOFT-FLOAT-NEXT: stg     %r0, 16(%r2)
; SOFT-FLOAT-NEXT: llihh   %r0, 16384
; SOFT-FLOAT-NEXT: stg     %r0, 8(%r2)
; SOFT-FLOAT-NEXT: llihh   %r0, 16368
; SOFT-FLOAT-NEXT: stg     %r0, 0(%r2)
; SOFT-FLOAT-NEXT: br      %r14
  ret { double, double, double, double, double }
      { double 1.0, double 2.0, double 3.0, double 4.0, double 5.0 }
}
