; RUN: not llvm-as < %s >/dev/null 2> %t
; RUN: FileCheck %s < %t
; Test the case where an invalid label name is used

; CHECK: invalid type for function argument

define void @test(label %bb) {
bb:
  ret void
}

