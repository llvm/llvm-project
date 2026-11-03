; RUN: llc -mtriple=x86_64-pc-linux-gnu -enable-new-pm -start-before=gc-lowering -stop-after=gc-lowering -verify-machineinstrs -o - %s | FileCheck %s

; Only @f is printed, even though no machine pass created its MF.
; CHECK-NOT: {{^}}name:
; CHECK: {{^}}name: f{{$}}
; CHECK-NOT: {{^}}name:

declare void @ext()

define available_externally void @ae() {
  ret void
}

define void @f() {
  call void @ext()
  ret void
}
