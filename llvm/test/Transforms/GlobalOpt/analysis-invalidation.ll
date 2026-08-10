; RUN: opt -passes='function(require<no-op-function>),globalopt' %s -debug-pass-manager -S 2>&1 | FileCheck %s

; CHECK: Clearing all analysis results for: f
; CHECK-NOT: @f

define internal void @f() {
  ret void
}
