; RUN: not --crash llc < %s -march=nvptx 2>&1 | FileCheck %s
; used to seqfault and now fails with an "Undefined external symbol"

; CHECK: LLVM ERROR: Undefined external symbol "__powidf2"
define double @powi(double, i32) {
  %a = call double @llvm.powi.f64.i32(double %0, i32 %1)
  ret double %a
}

declare double @llvm.powi.f64.i32(double, i32) nounwind readnone
