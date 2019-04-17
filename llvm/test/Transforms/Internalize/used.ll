; RUN: opt < %s -internalize -S | FileCheck %s
; RUN: opt < %s -passes=internalize -S | FileCheck %s

@llvm.used = appending global [1 x void ()*] [void ()* @f], section "llvm.metadata"

@llvm.compiler.used = appending global [1 x void ()*] [void ()* @g], section "llvm.metadata"

; CHECK: define void @f()
define void @f() {
  ret void
}

; CHECK: define internal void @g()
define void @g() {
  ret void
}

; CHECK: define internal void @h()
define void @h() {
  ret void
}
