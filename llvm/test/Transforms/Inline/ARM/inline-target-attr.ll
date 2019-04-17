; RUN: opt < %s -mtriple=arm-unknown-linux-gnu -S -inline | FileCheck %s
; RUN: opt < %s -mtriple=arm-unknown-linux-gnu -S -passes='cgscc(inline)' | FileCheck %s
; Check that we only inline when we have compatible target attributes.
; ARM has implemented a target attribute that will verify that the attribute
; sets are compatible.

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

define i32 @thumb_fn() #2 {
entry:
  %call = call i32 @foo()
  ret i32 %call
; CHECK-LABEL: thumb_fn
; CHECK: call i32 @foo
}

define i32 @strict_align() #3 {
entry:
  %call = call i32 @foo()
  ret i32 %call
; CHECK-LABEL: strict_align
; CHECK: call i32 (...) @baz()
}

define i32 @soft_float_fn() #4 {
entry:
  %call = call i32 @foo()
  ret i32 %call
; CHECK-LABEL: soft_float_fn
; CHECK: call i32 @foo
}

attributes #0 = { "target-cpu"="generic" "target-features"="+dsp,+neon" }
attributes #1 = { "target-cpu"="generic" "target-features"="+dsp,+neon,+fp16" }
attributes #2 = { "target-cpu"="generic" "target-features"="+dsp,+neon,+fp16,+thumb-mode" }
attributes #3 = { "target-cpu"="generic" "target-features"="+dsp,+neon,+strict-align" }
attributes #4 = { "target-cpu"="generic" "target-features"="+dsp,+neon,+fp16,+soft-float" }
