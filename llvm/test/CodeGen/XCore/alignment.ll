; RUN: not llc < %s -march=xcore 2>&1 | FileCheck %s

; CHECK: emitPrologue unsupported alignment: 8
define void @f() nounwind {
entry:
  %BadAlignment = alloca i64, align 8
  ret void
}

