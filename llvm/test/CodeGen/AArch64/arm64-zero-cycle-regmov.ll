; RUN: llc -mtriple=arm64-apple-ios -mcpu=cyclone < %s | FileCheck %s
; rdar://12254953

define i32 @t(i32 %a, i32 %b, i32 %c, i32 %d) nounwind ssp {
entry:
; CHECK-LABEL: t:
; CHECK: mov x0, [[REG1:x[0-9]+]]
; CHECK: mov x1, [[REG2:x[0-9]+]]
; CHECK: bl _foo
; CHECK: mov x0, [[REG1]]
; CHECK: mov x1, [[REG2]]
  %call = call i32 @foo(i32 %c, i32 %d) nounwind
  %call1 = call i32 @foo(i32 %c, i32 %d) nounwind
  unreachable
}

declare i32 @foo(i32, i32)
