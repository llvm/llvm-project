; RUN: llvm-link -S -o - %p/pr22807.ll %p/Inputs/pr22807.ll 2>&1 | FileCheck %s

; CHECK: %struct.B = type { %struct.A }
; CHECK: %struct.A = type opaque
; CHECK: @g = external global %struct.B

%struct.B = type { %struct.A }
%struct.A = type opaque

@g = external global %struct.B

define ptr @test() {
  ret ptr @g
}
