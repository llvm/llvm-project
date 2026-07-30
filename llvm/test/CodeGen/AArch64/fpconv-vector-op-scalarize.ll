; RUN: llc < %s -mtriple=arm64-apple-darwin | FileCheck %s

; PR20778
; Check that the legalizer doesn't crash when scalarizing FP conversion
; instructions' operands.  The operands are all illegal on AArch64,
; ensuring they are legalized.  The results are all legal.

define <1 x double> @test_sitofp(<1 x i1> %in) {
; CHECK-LABEL: test_sitofp:
; CHECK:       sbfx  [[GPR:w[0-9]+]], w0, #0, #1
; CHECK-NEXT:  scvtf d0, [[GPR]]
; CHECK-NEXT:  ret
entry:
  %0 = sitofp <1 x i1> %in to <1 x double>
  ret <1 x double> %0
}

define <1 x double> @test_uitofp(<1 x i1> %in) {
; CHECK-LABEL: test_uitofp:
; CHECK:       and   [[GPR:w[0-9]+]], w0, #0x1
; CHECK-NEXT:  ucvtf d0, [[GPR]]
; CHECK-NEXT:  ret
entry:
  %0 = uitofp <1 x i1> %in to <1 x double>
  ret <1 x double> %0
}

define <1 x i64> @test_fptosi(<1 x fp128> %in) {
; CHECK-LABEL: test_fptosi:
; CHECK:       bl    ___fixtfdi
; CHECK-NEXT:  fmov  d0, x0
entry:
  %0 = fptosi <1 x fp128> %in to <1 x i64>
  ret <1 x i64> %0
}

define <1 x i64> @test_fptoui(<1 x fp128> %in) {
; CHECK-LABEL: test_fptoui:
; CHECK:       bl    ___fixunstfdi
; CHECK-NEXT:  fmov  d0, x0
entry:
  %0 = fptoui <1 x fp128> %in to <1 x i64>
  ret <1 x i64> %0
}
