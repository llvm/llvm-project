; RUN: llc < %s -mtriple=armv7-apple-ios -O0 | FileCheck %s

; rdar://12713765
; When realign-stack is set to false, make sure we are not creating stack
; objects that are assumed to be 64-byte aligned.

define void @test1(<16 x float>* noalias sret(<16 x float>) %agg.result) nounwind ssp "no-realign-stack" {
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
 %a1 = bitcast <16 x float>* %retval to float*
 %a2 = getelementptr inbounds float, float* %a1, i64 8
 %a3 = bitcast float* %a2 to <4 x float>*

 %b1 = bitcast <16 x float>* %agg.result to float*
 %b2 = getelementptr inbounds float, float* %b1, i64 8
 %b3 = bitcast float* %b2 to <4 x float>*

 %0 = load <4 x float>, <4 x float>* %a3, align 16
 %1 = load <4 x float>, <4 x float>* %b3, align 16
 store <4 x float> %0, <4 x float>* %b3, align 16
 store <4 x float> %1, <4 x float>* %a3, align 16
 ret void
}

define void @test2(<16 x float>* noalias sret(<16 x float>) %agg.result) nounwind ssp {
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
 %a1 = bitcast <16 x float>* %retval to float*
 %a2 = getelementptr inbounds float, float* %a1, i64 8
 %a3 = bitcast float* %a2 to <4 x float>*

 %b1 = bitcast <16 x float>* %agg.result to float*
 %b2 = getelementptr inbounds float, float* %b1, i64 8
 %b3 = bitcast float* %b2 to <4 x float>*

 %0 = load <4 x float>, <4 x float>* %a3, align 16
 %1 = load <4 x float>, <4 x float>* %b3, align 16
 store <4 x float> %0, <4 x float>* %b3, align 16
 store <4 x float> %1, <4 x float>* %a3, align 16
 ret void
}
