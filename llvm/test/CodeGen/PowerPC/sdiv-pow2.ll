; RUN: llc -verify-machineinstrs -mcpu=ppc64 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu -mcpu=ppc < %s | FileCheck -check-prefix=CHECK-32 %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readnone
define signext i32 @foo4(i32 signext %a) #0 {
entry:
  %div = sdiv i32 %a, 8
  ret i32 %div

; CHECK-LABEL: @foo4
; CHECK: srawi [[REG1:[0-9]+]], 3, 3
; CHECK: addze [[REG2:[0-9]+]], [[REG1]]
; CHECK: extsw 3, [[REG2]]
; CHECK: blr
}

; Function Attrs: nounwind readnone
define i64 @foo8(i64 %a) #0 {
entry:
  %div = sdiv i64 %a, 8
  ret i64 %div

; CHECK-LABEL: @foo8
; CHECK: sradi [[REG1:[0-9]+]], 3, 3
; CHECK: addze 3, [[REG1]]
; CHECK: blr

; CHECK-32-LABEL: @foo8
; CHECK-32-NOT: sradi
; CHECK-32: blr
}

; Function Attrs: nounwind readnone
define signext i32 @foo4n(i32 signext %a) #0 {
entry:
  %div = sdiv i32 %a, -8
  ret i32 %div

; CHECK-LABEL: @foo4n
; CHECK: srawi [[REG1:[0-9]+]], 3, 3
; CHECK: addze [[REG2:[0-9]+]], [[REG1]]
; CHECK: neg [[REG3:[0-9]+]], [[REG2]]
; CHECK: extsw 3, [[REG3]]
; CHECK: blr
}

; Function Attrs: nounwind readnone
define i64 @foo8n(i64 %a) #0 {
entry:
  %div = sdiv i64 %a, -8
  ret i64 %div

; CHECK-LABEL: @foo8n
; CHECK: sradi [[REG1:[0-9]+]], 3, 3
; CHECK: addze [[REG2:[0-9]+]], [[REG1]]
; CHECK: neg 3, [[REG2]]
; CHECK: blr

; CHECK-32-LABEL: @foo8n
; CHECK-32-NOT: sradi
; CHECK-32: blr
}

attributes #0 = { nounwind readnone }

