; Test hot function attribute
; RUN: llc < %s | FileCheck %s
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: hot noinline norecurse nounwind readnone uwtable
define dso_local i32 @hot4() #0 {
entry:
  ret i32 1
}
; CHECK: .section        .text.hot.,"ax",@progbits
; CHECK: .globl  hot4

attributes #0 = { hot noinline norecurse nounwind readnone uwtable }
