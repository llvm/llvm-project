; RUN: llc < %s -mtriple=armv7-apple-ios -O0 | FileCheck %s

; rdar://12713765
; When realign-stack is set to false, make sure we are not creating stack
; objects that are assumed to be 64-byte aligned.
@T3_retval = common global <16 x float> zeroinitializer, align 16

define void @test1(<16 x float>* noalias sret(<16 x float>) %agg.result) nounwind ssp "no-realign-stack" {
entry:
; CHECK-LABEL: test1:
; CHECK: ldr     r[[R1:[0-9]+]], [pc, r[[R1]]]
; CHECK: mov     r[[R2:[0-9]+]], r[[R1]]
; CHECK: vld1.32 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R2]]:128]!
; CHECK: vld1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R2]]:128]
; CHECK: add     r[[R3:[0-9]+]], r[[R1]], #32
; CHECK: vld1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R3]]:128]
; CHECK: add     r[[R3:[0-9]+]], r[[R1]], #48
; CHECK: vld1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R3]]:128]
; CHECK: mov     r[[R2:[0-9]+]], sp
; CHECK: add     r[[R3:[0-9]+]], r[[R2]], #48
; CHECK: vst1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R3]]:128]
; CHECK: add     r[[R4:[0-9]+]], r[[R2]], #32
; CHECK: vst1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R4]]:128]
; CHECK: mov     r[[R5:[0-9]+]], r[[R2]]
; CHECK: vst1.32 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R5]]:128]!
; CHECK: vst1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R5]]:128]
; CHECK: vld1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R5]]:128]
; CHECK: vld1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R2]]:128]
; CHECK: vld1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R4]]:128]
; CHECK: vld1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R3]]:128]
; CHECK: add     r[[R1:[0-9]+]], r0, #48
; CHECK: vst1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R1]]:128]
; CHECK: add     r[[R1:[0-9]+]], r0, #32
; CHECK: vst1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R1]]:128]
; CHECK: vst1.32 {{{d[0-9]+}}, {{d[0-9]+}}}, [r0:128]!
; CHECK: vst1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r0:128]
 %retval = alloca <16 x float>, align 64
 %0 = load <16 x float>, <16 x float>* @T3_retval, align 16
 store <16 x float> %0, <16 x float>* %retval
 %1 = load <16 x float>, <16 x float>* %retval
 store <16 x float> %1, <16 x float>* %agg.result, align 16
 ret void
}

define void @test2(<16 x float>* noalias sret(<16 x float>) %agg.result) nounwind ssp {
entry:
; CHECK-LABEL: test2:
; CHECK: ldr     r[[R1:[0-9]+]], [pc, r[[R1]]]
; CHECK: add     r[[R2:[0-9]+]], r[[R1]], #48
; CHECK: vld1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R2]]:128]
; CHECK: add     r[[R2:[0-9]+]], r[[R1]], #32
; CHECK: vld1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R2]]:128]
; CHECK: vld1.32 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R1]]:128]!
; CHECK: vld1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R1]]:128]
; CHECK: mov     r[[R1:[0-9]+]], sp
; CHECK: orr     r[[R2:[0-9]+]], r[[R1]], #16
; CHECK: vst1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R2]]:128]
; CHECK: mov     r[[R3:[0-9]+]], #32
; CHECK: mov     r[[R9:[0-9]+]], r[[R1]]
; CHECK: vst1.32 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R9]]:128], r[[R3]]
; CHECK: mov     r[[R3:[0-9]+]], r[[R9]]
; CHECK: vst1.32 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R3]]:128]!
; CHECK: vst1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R3]]:128]
; CHECK: vld1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R9]]:128]
; CHECK: vld1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R3]]:128]
; CHECK: vld1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R2]]:128]
; CHECK: vld1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R1]]:128]
; CHECK: add     r[[R1:[0-9]+]], r0, #48
; CHECK: vst1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R1]]:128]
; CHECK: add     r[[R1:[0-9]+]], r0, #32
; CHECK: vst1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r[[R1]]:128]
; CHECK: vst1.32 {{{d[0-9]+}}, {{d[0-9]+}}}, [r0:128]!
; CHECK: vst1.64 {{{d[0-9]+}}, {{d[0-9]+}}}, [r0:128]


%retval = alloca <16 x float>, align 64
 %0 = load <16 x float>, <16 x float>* @T3_retval, align 16
 store <16 x float> %0, <16 x float>* %retval
 %1 = load <16 x float>, <16 x float>* %retval
 store <16 x float> %1, <16 x float>* %agg.result, align 16
 ret void
}
