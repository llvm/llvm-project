; RUN: llc  -relocation-model=pic -mtriple=thumb-apple-darwin < %s | FileCheck %s

declare hidden void @f()

; CHECK: bl _f

define void @g() {
  call void @f()
  ret void
}
