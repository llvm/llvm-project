; RUN: llc < %s -mtriple=x86_64 | FileCheck %s

; CHECK:    movq $0, {{[-0-9]+}}(%r{{[sb]}}p) # 8-byte Folded Spill
define i32 @test() nounwind {
entry:
  %div = udiv i256 0, 0
  store i256 %div, ptr null, align 16
  ret i32 0
}
