; RUN: opt -S < %s | FileCheck %s

declare void @foo()

define internal void @bar() {
  call void @foo() readnone
  ret void
}
