; RUN: not llc < %s -march=nvptx 2>&1 | FileCheck %s
; used to seqfault and now fails with a "Cannot select"

; CHECK: LLVM ERROR: Cannot select: {{t7|0x[0-9a-f]+}}: i32 = ExternalSymbol'__powidf2'
define double @powi() {
  %1 = call double @llvm.powi.f64(double 1.000000e+00, i32 undef)
  ret double %1
}

declare double @llvm.powi.f64(double, i32) nounwind readnone
