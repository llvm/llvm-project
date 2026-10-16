; RUN: opt < %s -passes=globalopt -S | FileCheck %s

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

$foo = comdat any
$trulydead = comdat any

; CHECK: @foo
define internal void @foo() comdat {
  ret void
}

; CHECK: @bar
define internal void @bar() #0 comdat($foo) {
  ret void
}

define void @zed()  {
  call void @bar()
  ret void
}

; CHECK-NOT: @trulydead
define internal void @trulydead() comdat {
  ret void
}

; CHECK-NOT: @trulydead2
define internal void @trulydead2() comdat($trulydead) {
  ret void
}

define i32 @main() {
  ret i32 0
}

attributes #0 = { noinline }