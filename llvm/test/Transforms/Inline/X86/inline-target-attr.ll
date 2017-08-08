; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -S -inline | FileCheck %s
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -S -passes='cgscc(inline)' | FileCheck %s
; Check that we only inline when we have compatible target attributes.
; X86 has implemented a target attribute that will verify that the attribute
; sets are compatible.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

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

attributes #0 = { "target-cpu"="x86-64" "target-features"="+sse,+sse2" }
attributes #1 = { "target-cpu"="x86-64" "target-features"="+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3" }
