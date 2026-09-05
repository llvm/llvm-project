; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; CHECK: error: constant vector elements must begin with a type

define <2 x i16> @test(<2 x i16> %in) {
entry:
  %arst = add <2 x i16> %in,
      <2 x i16> <i16 123, i16 456>
  ret <2 x i16> %arst
}
