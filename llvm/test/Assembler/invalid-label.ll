; RUN: not llvm-as < %s >/dev/null 2> %t
; RUN: FileCheck %s < %t
; Test the case where an invalid label name is used

; CHECK: unable to create block named 'bb'

define void @test(label %bb) {
bb:
  ret void
}

