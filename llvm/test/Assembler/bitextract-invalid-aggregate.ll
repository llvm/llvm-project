; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: error: invalid bitextract operands

define void @test_extract_struct(b32 %src) {
  %res = bitextract { i8, i8 }, b32 %src, i32 0
  ret void
}
