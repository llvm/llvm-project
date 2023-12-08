; RUN: llc < %s -mtriple=arm64-apple-darwin -enable-misched=false -enable-post-misched=false | FileCheck %s

; rdar://12713765
; Make sure we are not creating stack objects that are assumed to be 64-byte
; aligned.
@T3_retval = common global <16 x float> zeroinitializer, align 16

define void @test(ptr noalias sret(<16 x float>) %agg.result) nounwind ssp {
entry:
; CHECK: test
; CHECK: stp [[Q1:q[0-9]+]], [[Q2:q[0-9]+]], [sp, #32]
; CHECK: stp [[Q1:q[0-9]+]], [[Q2:q[0-9]+]], [sp]
; CHECK: stp [[Q1:q[0-9]+]], [[Q2:q[0-9]+]], [[[BASE:x[0-9]+]], #32]
; CHECK: stp [[Q1:q[0-9]+]], [[Q2:q[0-9]+]], [[[BASE]]]
 %retval = alloca <16 x float>, align 16
 %0 = load <16 x float>, ptr @T3_retval, align 16
 store <16 x float> %0, ptr %retval
 %1 = load <16 x float>, ptr %retval
 store <16 x float> %1, ptr %agg.result, align 16
 ret void
}
