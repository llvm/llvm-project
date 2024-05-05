; RUN: opt -S -passes=normalize < %s | FileCheck %s

define void @foo() {
  ret void
}