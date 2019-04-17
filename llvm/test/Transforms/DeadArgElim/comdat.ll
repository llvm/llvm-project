; RUN: opt -S < %s -deadargelim | FileCheck %s

$f = comdat any

define void @f() comdat {
  call void @g(i32 0)
  ret void
}

define internal void @g(i32 %dead) comdat($f) {
  ret void
}

; CHECK: define internal void @g() comdat($f) {
