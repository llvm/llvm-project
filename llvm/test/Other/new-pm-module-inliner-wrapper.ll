; RUN: opt -disable-verify -debug-pass-manager -passes='always-inliner-wrapper' -S %s 2>&1 | FileCheck %s

; CHECK: Running pass: InlinerPass

define void @foo() {
  ret void
}
