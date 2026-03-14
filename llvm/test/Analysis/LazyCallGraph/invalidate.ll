; Make sure we invalidate lcg even when preserving domtree
; RUN: opt -passes='require<lcg>,function(instcombine)' -debug-pass-manager -disable-output < %s 2>&1 | FileCheck %s

; CHECK: Invalidating {{.*}} LazyCallGraphAnalysis

define void @f() {
lbl:
  %a = add i32 1, 2
  unreachable
}
