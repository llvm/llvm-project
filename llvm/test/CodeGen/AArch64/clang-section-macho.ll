;RUN: llc -mtriple=arm64-apple-ios %s -o - | FileCheck %s

define dso_local void @foo() #0 {
entry:
  ret void
}

attributes #0 = { "implicit-section-name"="__TEXT,__mytext" }

; CHECK:      .section	__TEXT,__mytext
; CHECK-NEXT: .globl	_foo


define dso_local void @bar() #1 {
entry:
  ret void
}

attributes #1 = { "implicit-section-name"="__EDATA,zerofill" }

; CHECK:      .section	__EDATA,zerofill
; CHECK-NEXT: .globl	_bar
