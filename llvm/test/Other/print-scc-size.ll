; RUN: opt -S -debug-pass-manager -passes=no-op-cgscc < %s 2>&1 | FileCheck %s

; CHECK: Running pass: NoOpCGSCCPass on (f) (1 node)
; CHECK: Running pass: NoOpCGSCCPass on (g, h) (2 nodes)

define void @f() {
  ret void
}

define void @g() {
  call void @h()
  ret void
}

define void @h() {
  call void @g()
  ret void
}
