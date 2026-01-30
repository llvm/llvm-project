; RUN: llc < %s -mtriple x86_64-unknown-linux-gnu -stack-usage-file=%t.su
; RUN: FileCheck --input-file=%t.su %s

declare void @g(ptr)

define void @f() {
  %a = alloca [64 x i8]
  call void @g(ptr %a)
  ret void
}

; CHECK: f {{[0-9]+}} static

