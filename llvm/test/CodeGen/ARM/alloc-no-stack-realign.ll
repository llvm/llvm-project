; RUN: llc < %s -mtriple=armv7-apple-ios -O0 | FileCheck %s

; rdar://12713765
; When realign-stack is set to false, make sure we are not creating stack
; objects that are assumed to be 64-byte aligned.

define void @test1(ptr noalias sret(<16 x float>) %agg.result) nounwind ssp "no-realign-stack" {
; CHECK-LABEL: test1:
; CHECK: mov r[[PTR:[0-9]+]], r{{[0-9]+}}
; CHECK: mov r[[NOTALIGNED:[0-9]+]], sp
; CHECK: add r[[NOTALIGNED]], r[[NOTALIGNED]], #32
; CHECK: add r[[PTR]], r[[PTR]], #32
; CHECK: vld1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[NOTALIGNED]]:128]
; CHECK: vld1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[PTR]]:128]
; CHECK: vst1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[PTR]]:128]
; CHECK: vst1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[NOTALIGNED]]:128]
entry:
 %retval = alloca <16 x float>, align 64
 %a2 = getelementptr inbounds float, ptr %retval, i64 8

 %b2 = getelementptr inbounds float, ptr %agg.result, i64 8

 %0 = load <4 x float>, ptr %a2, align 16
 %1 = load <4 x float>, ptr %b2, align 16
 store <4 x float> %0, ptr %b2, align 16
 store <4 x float> %1, ptr %a2, align 16
 ret void
}

define void @test2(ptr noalias sret(<16 x float>) %agg.result) nounwind ssp {
; CHECK-LABEL: test2:
; CHECK: mov r[[PTR:[0-9]+]], r{{[0-9]+}}
; CHECK: mov r[[ALIGNED:[0-9]+]], sp
; CHECK: orr r[[ALIGNED]], r[[ALIGNED]], #32
; CHECK: add r[[PTR]], r[[PTR]], #32
; CHECK: vld1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[ALIGNED]]:128]
; CHECK: vld1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[PTR]]:128]
; CHECK: vst1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[PTR]]:128]
; CHECK: vst1.64 {d{{[0-9]+}}, d{{[0-9]+}}}, [r[[ALIGNED]]:128]
entry:
 %retval = alloca <16 x float>, align 64
 %a2 = getelementptr inbounds float, ptr %retval, i64 8

 %b2 = getelementptr inbounds float, ptr %agg.result, i64 8

 %0 = load <4 x float>, ptr %a2, align 16
 %1 = load <4 x float>, ptr %b2, align 16
 store <4 x float> %0, ptr %b2, align 16
 store <4 x float> %1, ptr %a2, align 16
 ret void
}
