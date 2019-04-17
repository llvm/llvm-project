; RUN: opt < %s -mtriple=aarch64-unknown-linux-gnu -S -inline | FileCheck %s
; RUN: opt < %s -mtriple=aarch64-unknown-linux-gnu -S -passes='cgscc(inline)' | FileCheck %s
; Check that we only inline when we have compatible target attributes.

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

define i32 @strict_align() #2 {
entry:
  %call = call i32 @foo()
  ret i32 %call
; CHECK-LABEL: strict_align
; CHECK: call i32 (...) @baz()
}

attributes #0 = { "target-cpu"="generic" "target-features"="+crc,+neon" }
attributes #1 = { "target-cpu"="generic" "target-features"="+crc,+neon,+crypto" }
attributes #2 = { "target-cpu"="generic" "target-features"="+crc,+neon,+strict-align" }
