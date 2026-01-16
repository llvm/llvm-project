; RUN: opt < %s -mtriple=arm-unknown-linux-gnu -S -passes=inline | FileCheck %s
; RUN: opt < %s -mtriple=arm-unknown-linux-gnu -S -passes='cgscc(inline)' | FileCheck %s

declare i32 @foo(...) #0

define i32 @callee() #0 {
entry:
  %call = call i32 (...) @foo()
  ret i32 %call
}

define i32 @dotcallee() #1 {
entry:
  %call = call i32 (...) @foo()
  ret i32 %call
}

define i32 @dotcaller() #1 {
entry:
  %call = call i32 @callee()
  ret i32 %call
; CHECK-LABEL: dotcaller
; CHECK: call i32 (...) @foo()
}

define i32 @caller() #0 {
entry:
  %call = call i32 @dotcallee()
  ret i32 %call
; CHECK-LABEL: caller
; CHECK: call i32 @dotcallee()
}

attributes #0 = { "target-cpu"="generic" "target-features"="+dsp,+neon" }
attributes #1 = { "target-cpu"="generic" "target-features"="+dsp,+neon,+dotprod" }
