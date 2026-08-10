; RUN: llc < %s -mtriple=arm64-apple-darwin -O0 -fast-isel | FileCheck %s

; CHECK: b _foo0

define i32 @foo1() {
entry:
  %call = tail call i32 @foo0()
  ret i32 %call
}

declare i32 @foo0()
