;RUN: llc -mtriple=aarch64-unknown-unknown -o - -global-isel -global-isel-abort=2 %s 2>&1 | FileCheck %s
; CHECK: fallback
; CHECK-LABEL: foo
define i16 @foo(half* %p) {
  %tmp0 = load half, half* %p
  %tmp1 = fptoui half %tmp0 to i16
  ret i16 %tmp1
}
