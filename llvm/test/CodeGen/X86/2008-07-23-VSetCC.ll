; RUN: llc < %s -march=x86 -mcpu=pentium
; PR2575

define void @entry(i32 %m_task_id, i32 %start_x, i32 %end_x) nounwind  {
	br i1 false, label %bb.nph, label %._crit_edge

bb.nph:		; preds = %bb.nph, %0
	%X = icmp sgt <4 x i32> zeroinitializer, < i32 -128, i32 -128, i32 -128, i32 -128 >		; <<4 x i32>>:1 [#uses=1]
        sext <4 x i1> %X to <4 x i32>
	extractelement <4 x i32> %1, i32 3		; <i32>:2 [#uses=1]
	lshr i32 %2, 31		; <i32>:3 [#uses=1]
	trunc i32 %3 to i1		; <i1>:4 [#uses=1]
	select i1 %4, i32 -1, i32 0		; <i32>:5 [#uses=1]
	insertelement <4 x i32> zeroinitializer, i32 %5, i32 3		; <<4 x i32>>:6 [#uses=1]
	and <4 x i32> zeroinitializer, %6		; <<4 x i32>>:7 [#uses=1]
	bitcast <4 x i32> %7 to <4 x float>		; <<4 x float>>:8 [#uses=1]
	fmul <4 x float> zeroinitializer, %8		; <<4 x float>>:9 [#uses=1]
	bitcast <4 x float> %9 to <4 x i32>		; <<4 x i32>>:10 [#uses=1]
	or <4 x i32> %10, zeroinitializer		; <<4 x i32>>:11 [#uses=1]
	bitcast <4 x i32> %11 to <4 x float>		; <<4 x float>>:12 [#uses=1]
	fmul <4 x float> %12, < float 1.000000e+02, float 1.000000e+02, float 1.000000e+02, float 1.000000e+02 >		; <<4 x float>>:13 [#uses=1]
	fsub <4 x float> %13, < float 1.000000e+02, float 1.000000e+02, float 1.000000e+02, float 1.000000e+02 >		; <<4 x float>>:14 [#uses=1]
	extractelement <4 x float> %14, i32 3		; <float>:15 [#uses=1]
	call float @fmaxf( float 0.000000e+00, float %15 )		; <float>:16 [#uses=0]
	br label %bb.nph

._crit_edge:		; preds = %0
	ret void
}


declare float @fmaxf(float, float)
