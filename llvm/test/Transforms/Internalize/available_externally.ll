; RUN: opt < %s -internalize -internalize-public-api-list foo -S | FileCheck %s

; CHECK: define void @foo
define void @foo() {
  ret void
}

; CHECK: define internal void @zed
define void @zed() {
  ret void
}

; CHECK: define available_externally void @bar
define available_externally void @bar() {
  ret void
}
