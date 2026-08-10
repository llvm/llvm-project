; RUN: opt < %s -passes=instcombine

define void @entry(i32 %m_task_id, i32 %start_x, i32 %end_x, i32 %start_y, i32 %end_y) {
	br label %1

; <label>:1		; preds = %4, %0
	%2 = icmp slt i32 0, %end_y		; <i1> [#uses=1]
	br i1 %2, label %4, label %3

; <label>:3		; preds = %1
	ret void

; <label>:4		; preds = %6, %1
	%5 = icmp slt i32 0, %end_x		; <i1> [#uses=1]
	br i1 %5, label %6, label %1

; <label>:6		; preds = %4
	%7 = srem <2 x i32> zeroinitializer, zeroinitializer		; <<2 x i32>> [#uses=1]
	%8 = extractelement <2 x i32> %7, i32 1		; <i32> [#uses=1]
	%9 = select i1 false, i32 0, i32 %8		; <i32> [#uses=1]
	%10 = insertelement <2 x i32> zeroinitializer, i32 %9, i32 1		; <<2 x i32>> [#uses=1]
	%11 = extractelement <2 x i32> %10, i32 1		; <i32> [#uses=1]
	%12 = insertelement <4 x i32> zeroinitializer, i32 %11, i32 3		; <<4 x i32>> [#uses=1]
	%13 = sitofp <4 x i32> %12 to <4 x float>		; <<4 x float>> [#uses=1]
	store <4 x float> %13, ptr null
	br label %4
}
