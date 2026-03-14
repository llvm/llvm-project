; Check that AArch64 is honoring code-model=large at -O0 and -O2.
;
; RUN: llc -mtriple=arm64-apple-darwin19 -code-model=large -O0 -o - %s | FileCheck %s
; RUN: llc -mtriple=arm64-apple-darwin19 -code-model=large -O2 -o - %s | FileCheck %s

; CHECK: adrp    [[REG1:x[0-9]+]], _bar@GOTPAGE
; CHECK: ldr     [[REG1]], [[[REG1]], _bar@GOTPAGEOFF]
; CHECK: blr     [[REG1]]

declare void @bar()

define void @foo() {
  call void @bar()
  ret void
}

