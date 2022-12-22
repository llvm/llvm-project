; RUN: llc < %s -mtriple=aarch64--linux-gnueabihf | FileCheck %s

;; This used to cause a backend crash about not being able to
;; select ROTL. Make sure if generates the basic ushr/shl.
define <2 x i64> @testcase(ptr %in) {
; CHECK-LABEL: testcase
; CHECK: ushr {{v[0-9]+}}.2d
; CHECK: shl  {{v[0-9]+}}.2d
  %1 = load <2 x i64>, ptr %in
  %2 = lshr <2 x i64> %1, <i64 8, i64 8>
  %3 = shl <2 x i64> %1, <i64 56, i64 56>
  %4 = or <2 x i64> %2, %3
  ret <2 x i64> %4
}
