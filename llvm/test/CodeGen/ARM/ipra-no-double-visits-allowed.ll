; Regression test for the orphan catch-up loop in CGPassManager (see
; https://github.com/llvm/llvm-project/issues/119556). The SCC traversal that
; starts at an orphan must not run the CGSCC pipeline again on functions that
; were already visited, either by the main traversal (@shared, reached from
; @foo) or by an earlier orphan traversal (@orphan_leaf, an orphan itself that
; is defined before its orphan caller). Visiting them twice emits their bodies
; twice and fails with "symbol 'shared' is already defined".

; RUN: llc -mtriple=armv7-unknown-linux -O0 -enable-ipra < %s \
; RUN:   | FileCheck %s --implicit-check-not={{^}}shared: \
; RUN:       --implicit-check-not={{^}}orphan_leaf:
; RUN: llc -mtriple=armv7-unknown-linux -enable-ipra < %s \
; RUN:   | FileCheck %s --implicit-check-not={{^}}shared: \
; RUN:       --implicit-check-not={{^}}orphan_leaf:
; RUN: llc -mtriple=thumbv7-unknown-linux -O0 -enable-ipra < %s \
; RUN:   | FileCheck %s --implicit-check-not={{^}}shared: \
; RUN:       --implicit-check-not={{^}}orphan_leaf:
; RUN: llc -mtriple=thumbv7-unknown-linux -enable-ipra < %s \
; RUN:   | FileCheck %s --implicit-check-not={{^}}shared: \
; RUN:       --implicit-check-not={{^}}orphan_leaf:

; CHECK-DAG: {{^}}shared:
; CHECK-DAG: {{^}}foo:
; CHECK-DAG: {{^}}orphan:
; CHECK-DAG: {{^}}orphan_leaf:
; CHECK-DAG: {{^}}orphan_caller:

define void @shared() {
  ret void
}

define void @foo() {
  call void @shared()
  ret void
}

define internal void @orphan() {
  call void @shared()
  ret void
}

define internal void @orphan_leaf() {
  ret void
}

define internal void @orphan_caller() {
  call void @orphan_leaf()
  ret void
}
