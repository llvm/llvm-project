target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$foo = comdat any
define linkonce void @foo() comdat {
  ret void
}

define void @bar() {
  call void @foo()
  ret void
}
