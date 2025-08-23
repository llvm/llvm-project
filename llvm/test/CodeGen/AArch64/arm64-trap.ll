; RUN: llc < %s -mtriple=arm64-eabi | FileCheck %s
; RUN: llc < %s -mtriple=arm64-eabi -global-isel | FileCheck %s

define void @foo() nounwind {
; CHECK-LABEL: foo
; CHECK: brk #0x1
  tail call void @llvm.trap()
  ret void
}
declare void @llvm.trap() nounwind

; CHECK-LABEL: {{\_?}}foo_trap_func:
; CHECK: bl	trap_func

define void @foo_trap_func() {
  call void @llvm.trap() #0
  unreachable
}

attributes #0 = { "trap-func-name"="trap_func" }
