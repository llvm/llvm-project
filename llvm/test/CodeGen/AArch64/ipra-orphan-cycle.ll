; Regression test for the cyclic-orphan case of the IPRA empty-MachineFunction
; bug (see https://github.com/llvm/llvm-project/issues/119556). Two mutually
; recursive `internal` functions with no external entry point: neither is
; use_empty, neither has external linkage, neither is address-taken, but the
; 2-node SCC {foo, bar} has no incoming edge from
; ExternalCallingNode and is therefore not visited by scc_iterator.

; RUN: llc -mtriple=aarch64    -enable-ipra < %s | FileCheck %s
; RUN: llc -mtriple=aarch64_be -enable-ipra < %s | FileCheck %s

; CHECK-LABEL: foo:
; CHECK: ret
; CHECK-LABEL: bar:
; CHECK: ret

define internal void @foo() {
  call void @bar()
  ret void
}

define internal void @bar() {
  call void @foo()
  ret void
}
