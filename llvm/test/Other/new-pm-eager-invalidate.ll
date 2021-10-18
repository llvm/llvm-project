; RUN: opt -disable-verify -debug-pass-manager -passes='function(require<no-op-function>)' -disable-output -eagerly-invalidate-analyses %s 2>&1 | FileCheck %s
; RUN: opt -disable-verify -debug-pass-manager -passes='cgscc(function(require<no-op-function>))' -disable-output -eagerly-invalidate-analyses %s 2>&1 | FileCheck %s

; CHECK: Invalidating analysis: NoOpFunctionAnalysis

define void @foo() {
  unreachable
}
