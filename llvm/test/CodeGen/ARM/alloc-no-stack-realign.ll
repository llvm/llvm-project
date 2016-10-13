; RUN: llc < %s -mtriple=armv7-apple-ios -O0 | FileCheck %s -check-prefix=NO-REALIGN
; RUN: llc < %s -mtriple=armv7-apple-ios -O0 | FileCheck %s -check-prefix=REALIGN

; rdar://12713765
; When realign-stack is set to false, make sure we are not creating stack
; objects that are assumed to be 64-byte aligned.
@T3_retval = common global <16 x float> zeroinitializer, align 16

define void @test1(<16 x float>* noalias sret %agg.result) nounwind ssp "no-realign-stack" {
entry:
; NO-REALIGN-LABEL: test1
; NO-REALIGN: mov r[[R2:[0-9]+]], r[[R1:[0-9]+]]
; NO-REALIGN: vld1.32 {{{d[0-9]+, d[0-9]+}}}, [r[[R2]]:128]!
; NO-REALIGN: vld1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R2]]:128]
; NO-REALIGN: add r[[R2:[0-9]+]], r[[R1]], #32
; NO-REALIGN: vld1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R2]]:128]
; NO-REALIGN: add r[[R2:[0-9]+]], r[[R1]], #48
; NO-REALIGN: vld1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R2]]:128]

; NO-REALIGN: add r[[R2:[0-9]+]], r[[R1:[0-9]+]], #48
; NO-REALIGN: vst1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R2]]:128]
; NO-REALIGN: add r[[R2:[0-9]+]], r[[R1]], #32
; NO-REALIGN: vst1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R2]]:128]
; NO-REALIGN: mov	r[[R3:[0-9]+]], r[[R1]]
; NO-REALIGN: vst1.32 {{{d[0-9]+, d[0-9]+}}}, [r[[R3]]:128]!
; NO-REALIGN: vst1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R3]]:128]

; NO-REALIGN: add r[[R2:[0-9]+]], r[[R0:0]], #48
; NO-REALIGN: vst1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R2]]:128]
; NO-REALIGN: add r[[R2:[0-9]+]], r[[R0]], #32
; NO-REALIGN: vst1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R2]]:128]
; NO-REALIGN: vst1.32 {{{d[0-9]+, d[0-9]+}}}, [r[[R0]]:128]!
; NO-REALIGN: vst1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R0]]:128]
 %retval = alloca <16 x float>, align 16
 %0 = load <16 x float>, <16 x float>* @T3_retval, align 16
 store <16 x float> %0, <16 x float>* %retval
 %1 = load <16 x float>, <16 x float>* %retval
 store <16 x float> %1, <16 x float>* %agg.result, align 16
 ret void
}

define void @test2(<16 x float>* noalias sret %agg.result) nounwind ssp {
entry:
; REALIGN-LABEL: test2
; REALIGN: bfc sp, #0, #6
; REALIGN: mov r[[R2:[0-9]+]], r[[R1:[0-9]+]]
; REALIGN: vld1.32 {{{d[0-9]+, d[0-9]+}}}, [r[[R2]]:128]!
; REALIGN: vld1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R2]]:128]
; REALIGN: add r[[R2:[0-9]+]], r[[R1]], #32
; REALIGN: vld1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R2]]:128]
; REALIGN: add r[[R2:[0-9]+]], r[[R1]], #48
; REALIGN: vld1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R2]]:128]


; REALIGN: orr r[[R2:[0-9]+]], r[[R1:[0-9]+]], #48
; REALIGN: vst1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R2]]:128]
; REALIGN: orr r[[R2:[0-9]+]], r[[R1]], #32
; REALIGN: vst1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R2]]:128]
; REALIGN: orr r[[R2:[0-9]+]], r[[R1]], #16
; REALIGN: vst1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R2]]:128]
; REALIGN: vst1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R1]]:128]

; REALIGN: add r[[R1:[0-9]+]], r[[R0:0]], #48
; REALIGN: vst1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R1]]:128]
; REALIGN: add r[[R1:[0-9]+]], r[[R0]], #32
; REALIGN: vst1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R1]]:128]
; REALIGN: vst1.32 {{{d[0-9]+, d[0-9]+}}}, [r[[R0]]:128]!
; REALIGN: vst1.64 {{{d[0-9]+, d[0-9]+}}}, [r[[R0]]:128]
 %retval = alloca <16 x float>, align 16
 %0 = load <16 x float>, <16 x float>* @T3_retval, align 16
 store <16 x float> %0, <16 x float>* %retval
 %1 = load <16 x float>, <16 x float>* %retval
 store <16 x float> %1, <16 x float>* %agg.result, align 16
 ret void
}
