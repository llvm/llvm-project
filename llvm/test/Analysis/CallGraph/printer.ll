; RUN: opt -S -passes=print-callgraph-sccs -disable-output < %s 2>&1 | FileCheck %s
; CHECK: SCC #1: g, f
; CHECK: SCC #2: h
; CHECK: SCC #3: external node

define void @f() {
  call void @g()
  ret void
}

define void @g() {
  call void @f()
  ret void
}

define void @h() {
  call void @f()
  ret void
}
