; RUN: llc < %s -mtriple=armv7-apple-ios -O0 | FileCheck %s -check-prefix=NO-REALIGN
; RUN: llc < %s -mtriple=armv7-apple-ios -O0 | FileCheck %s -check-prefix=REALIGN

; rdar://12713765
; When realign-stack is set to false, make sure we are not creating stack
; objects that are assumed to be 64-byte aligned.
@T3_retval = common global <16 x float> zeroinitializer, align 16

define void @test1(<16 x float>* noalias sret %agg.result) nounwind ssp "no-realign-stack" {
entry:
; NO-REALIGN-LABEL: test1:
; NO-REALIGN: ldr	r1, [pc, r1]
; NO-REALIGN: mov	r2, r1
; NO-REALIGN: vld1.32	{d16, d17}, [r2:128]!
; NO-REALIGN: vld1.64	{d18, d19}, [r2:128]
; NO-REALIGN: add	r2, r1, #32
; NO-REALIGN: vld1.64	{d20, d21}, [r2:128]
; NO-REALIGN: add	r1, r1, #48
; NO-REALIGN: vld1.64	{d22, d23}, [r1:128]
; NO-REALIGN: mov	r1, sp
; NO-REALIGN: add	r2, r1, #48
; NO-REALIGN: vst1.64	{d22, d23}, [r2:128]
; NO-REALIGN: add	r3, r1, #32
; NO-REALIGN: vst1.64	{d20, d21}, [r3:128]
; NO-REALIGN: mov	r9, r1
; NO-REALIGN: vst1.32	{d16, d17}, [r9:128]!
; NO-REALIGN: vst1.64	{d18, d19}, [r9:128]
; NO-REALIGN: vld1.64	{d16, d17}, [r9:128]
; NO-REALIGN: vld1.64	{d18, d19}, [r1:128]
; NO-REALIGN: vld1.64	{d20, d21}, [r3:128]
; NO-REALIGN: vld1.64	{d22, d23}, [r2:128]
; NO-REALIGN: add	r1, r0, #48
; NO-REALIGN: vst1.64	{d22, d23}, [r1:128]
; NO-REALIGN: add	r1, r0, #32
; NO-REALIGN: vst1.64	{d20, d21}, [r1:128]
; NO-REALIGN: vst1.32	{d18, d19}, [r0:128]!
; NO-REALIGN: vst1.64	{d16, d17}, [r0:128]
 %retval = alloca <16 x float>, align 16
 %0 = load <16 x float>, <16 x float>* @T3_retval, align 16
 store <16 x float> %0, <16 x float>* %retval
 %1 = load <16 x float>, <16 x float>* %retval
 store <16 x float> %1, <16 x float>* %agg.result, align 16
 ret void
}

define void @test2(<16 x float>* noalias sret %agg.result) nounwind ssp {
entry:
; NO-REALIGN-LABEL: test2:
; NO-REALIGN: ldr	r1, [pc, r1]
; NO-REALIGN: add	r2, r1, #48
; NO-REALIGN: vld1.64	{d16, d17}, [r2:128]
; NO-REALIGN: add	r2, r1, #32
; NO-REALIGN: vld1.64	{d18, d19}, [r2:128]
; NO-REALIGN: vld1.32	{d20, d21}, [r1:128]!
; NO-REALIGN: vld1.64	{d22, d23}, [r1:128]
; NO-REALIGN: mov	r1, sp
; NO-REALIGN: orr	r2, r1, #16
; NO-REALIGN: vst1.64	{d22, d23}, [r2:128]
; NO-REALIGN: mov	r3, #32
; NO-REALIGN: mov	r9, r1
; NO-REALIGN: vst1.32	{d20, d21}, [r9:128], r3
; NO-REALIGN: mov	r3, r9
; NO-REALIGN: vst1.32	{d18, d19}, [r3:128]!
; NO-REALIGN: vst1.64	{d16, d17}, [r3:128]
; NO-REALIGN: vld1.64	{d16, d17}, [r9:128]
; NO-REALIGN: vld1.64	{d18, d19}, [r3:128]
; NO-REALIGN: vld1.64	{d20, d21}, [r2:128]
; NO-REALIGN: vld1.64	{d22, d23}, [r1:128]
; NO-REALIGN: add	r1, r0, #48
; NO-REALIGN: vst1.64	{d18, d19}, [r1:128]
; NO-REALIGN: add	r1, r0, #32
; NO-REALIGN: vst1.64	{d16, d17}, [r1:128]
; NO-REALIGN: vst1.32	{d22, d23}, [r0:128]!
; NO-REALIGN: vst1.64	{d20, d21}, [r0:128]

; REALIGN: test2
 %retval = alloca <16 x float>, align 16
 %0 = load <16 x float>, <16 x float>* @T3_retval, align 16
 store <16 x float> %0, <16 x float>* %retval
 %1 = load <16 x float>, <16 x float>* %retval
 store <16 x float> %1, <16 x float>* %agg.result, align 16
 ret void
}
