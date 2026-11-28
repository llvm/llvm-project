; Regression test for the cyclic-orphan case of the IPRA empty-MachineFunction
; bug (see https://github.com/llvm/llvm-project/issues/119556). Two mutually
; recursive `internal` functions form an SCC unreachable from
; ExternalCallingNode. The CGPassManager catch-up loop must visit it.

; RUN: llc -mtriple=armv7-unknown-linux   -enable-ipra < %s | FileCheck %s
; RUN: llc -mtriple=armv7eb-unknown-linux -enable-ipra < %s | FileCheck %s
; RUN: llc -mtriple=thumbv7-unknown-linux -enable-ipra < %s | FileCheck %s
; RUN: llc -mtriple=thumbv8-unknown-linux -enable-ipra < %s | FileCheck %s

; CHECK-LABEL: foo:
; CHECK: pop
; CHECK-LABEL: bar:
; CHECK: pop

define internal void @foo() {
  call void @bar()
  ret void
}

define internal void @bar() {
  call void @foo()
  ret void
}
