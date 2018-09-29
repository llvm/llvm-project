; RUN: llc < %s -mtriple=arm64-apple-ios -mattr=-zcm   | FileCheck %s -check-prefixes=CHECK,NOT
; RUN: llc < %s -mtriple=arm64-apple-ios -mattr=+zcm   | FileCheck %s -check-prefixes=CHECK,YES
; RUN: llc < %s -mtriple=arm64-apple-ios -mcpu=cyclone | FileCheck %s -check-prefixes=CHECK,YES

; rdar://12254953
define i32 @t(i32 %a, i32 %b, i32 %c, i32 %d) nounwind ssp {
entry:
; CHECK-LABEL: t:
; NOT: mov [[REG2:w[0-9]+]], w3
; NOT: mov [[REG1:w[0-9]+]], w2
; YES: mov [[REG2:x[0-9]+]], x3
; YES: mov [[REG1:x[0-9]+]], x2
; CHECK: bl _foo
; NOT: mov w0, [[REG1]]
; NOT: mov w1, [[REG2]]
; YES: mov x0, [[REG1]]
; YES: mov x1, [[REG2]]
  %call = call i32 @foo(i32 %c, i32 %d) nounwind
  %call1 = call i32 @foo(i32 %c, i32 %d) nounwind
  unreachable
}

declare i32 @foo(i32, i32)
