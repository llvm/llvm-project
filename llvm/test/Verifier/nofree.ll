; RUN: not opt -S < %s 2>&1 | FileCheck %s

; CHECK: Attribute 'nofree' applied to incompatible type!
define void @test(i32 nofree %p) {
  ret void
}

; CHECK: Attribute 'nofreeobj' applied to incompatible type!
define void @test2(i32 nofreeobj %p) {
  ret void
}

; CHECK: Attribute 'nofreeobj' applied to incompatible type!
define nofreeobj i32 @test3() {
  ret i32 0
}
