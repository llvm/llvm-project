; RUN: llc < %s -march=cellspu | FileCheck %s

; ModuleID = 'loads.bc'
target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

define <4 x float> @load_v4f32_1(<4 x float>* %a) nounwind readonly {
entry:
	%tmp1 = load <4 x float>* %a
	ret <4 x float> %tmp1
; CHECK:	lqd	$3, 0($3)
}

define <4 x float> @load_v4f32_2(<4 x float>* %a) nounwind readonly {
entry:
	%arrayidx = getelementptr <4 x float>* %a, i32 1
	%tmp1 = load <4 x float>* %arrayidx
	ret <4 x float> %tmp1
; CHECK:	lqd	$3, 16($3)
}


declare <4 x i32>* @getv4f32ptr()
define <4 x i32> @func() {
	;CHECK: brasl
	; we need to have some instruction to move the result to safety.
	; which instruction (lr, stqd...) depends on the regalloc
	;CHECK: {{.*}}
	;CHECK: brasl
	%rv1 = call <4 x i32>* @getv4f32ptr()
	%rv2 = call <4 x i32>* @getv4f32ptr()
	%rv3 = load <4 x i32>* %rv1
	ret <4 x i32> %rv3
}

define <4 x float> @load_undef(){
	; CHECK: lqd	$3, 0($3)
	%val = load <4 x float>* undef
	ret <4 x float> %val
}

;check that 'misaligned' loads that may span two memory chunks
;have two loads. Don't check for the bitmanipulation, as that 
;might change with improved algorithms or scheduling 
define i32 @load_misaligned( i32* %ptr ){
;CHECK: load_misaligned
;CHECK: lqd
;CHECK: lqd
;CHECK: bi $lr
  %rv = load i32* %ptr, align 2
  ret i32 %rv
}

define <4 x i32> @load_null_vec( ) {
;CHECK: lqa
;CHECK: bi $lr
	%rv = load <4 x i32>* null
	ret <4 x i32> %rv
}
