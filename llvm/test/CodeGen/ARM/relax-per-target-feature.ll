; RUN: llc -mtriple=thumbv4t-linux-gnueabi -o - %s | FileCheck %s

; Functions may have more features than the base triple; code generation and
; instruction selection may be performed based on this information. This test
; makes sure that the MC layer performs instruction relaxation based on the
; target-features of the function. The relaxation for tail call is particularly
; important on Thumb2 as the 16-bit Thumb branch instruction has an extremely
; short range.

declare dso_local void @g(...) local_unnamed_addr #2

define dso_local void @f() local_unnamed_addr #0 {
entry:
  tail call void bitcast (void (...)* @g to void ()*)() #3
  ret void
}
; Function has thumb2 target-feature, tail call is allowed and must be widened.
; CHECK: f:
; CHECK: b g

define dso_local void @h() local_unnamed_addr #2 {
entry:
  tail call void bitcast (void (...)* @g to void ()*)() #3
  ret void
}
; Function does not have thumb2 target-feature, tail call should not be
; generated as it cannot be widened.
; CHECK: h:
; CHECK: bl g

attributes #0 = { nounwind  "disable-tail-calls"="false" "target-cpu"="cortex-a53" "target-features"="+crypto,+fp-armv8,+neon,+soft-float-abi,+strict-align,+thumb-mode,-crc,-dotprod,-dsp,-hwdiv,-hwdiv-arm,-ras" "use-soft-float"="true" }

attributes #2 = { nounwind  "disable-tail-calls"="false" "target-cpu"="arm7tdmi" "target-features"="+strict-align,+thumb-mode,-crc,-dotprod,-dsp,-hwdiv,-hwdiv-arm,-ras" "unsafe-fp-math"="false" "use-soft-float"="true" }
attributes #3 = { nounwind }
