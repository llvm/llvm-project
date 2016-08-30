; RUN: llc < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @bar() local_unnamed_addr #0 {
entry:
  tail call void @foo() #1
  ret void

; CHECK-LABEL: @bar
; CHECK: ld [[FD:[0-9]+]], .LC0@toc@l({{[0-9]+}})
; CHECK: ld [[ADDR:[0-9]+]], 0([[FD]])
; CHECK: mtctr [[ADDR]]
; CHECK: bctrl
; CHECK-NOT: bl foo
; CHECK: blr
}

; CHECK: .tc foo

declare void @foo() local_unnamed_addr

attributes #0 = { nounwind "target-cpu"="ppc64" "target-features"="+longcall" }
attributes #1 = { nounwind }

