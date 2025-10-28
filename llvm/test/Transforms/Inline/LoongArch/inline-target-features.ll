; RUN: opt < %s -mtriple=loongarch64-unknown-linux-gnu -S -passes=inline | FileCheck %s
; RUN: opt < %s -mtriple=loongarch64-unknown-linux-gnu -S -passes='cgscc(inline)' | FileCheck %s
; Check that we only inline when we have compatible target attributes.

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "loongarch64-unknown-linux-gnu"

define i32 @foo() #0 {
entry:
  %call = call i32 (...) @baz()
  ret i32 %call
; CHECK-LABEL: foo
; CHECK: call i32 (...) @baz()
}
declare i32 @baz(...) #0

define i32 @bar() #1 {
entry:
  %call = call i32 @foo()
  ret i32 %call
; CHECK-LABEL: bar
; CHECK: call i32 (...) @baz()
}

define i32 @qux() #0 {
entry:
  %call = call i32 @bar()
  ret i32 %call
; CHECK-LABEL: qux
; CHECK: call i32 @bar()
}

attributes #0 = { "target-cpu"="generic-la64" "target-features"="+f,+d" }
attributes #1 = { "target-cpu"="generic-la64" "target-features"="+f,+d,+lsx,+lasx" }
