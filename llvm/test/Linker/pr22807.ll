; RUN: not llvm-link -S -o - %p/pr22807.ll %p/Inputs/pr22807.ll 2>&1 | FileCheck %s

; CHECK: error: identified structure type 'struct.A' is recursive

%struct.B = type { %struct.A }
%struct.A = type opaque

@g = external global %struct.B

define ptr @test() {
  ret ptr @g
}
