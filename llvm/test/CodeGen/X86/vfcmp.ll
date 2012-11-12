; RUN: llc < %s -march=x86 -mattr=+sse2
; PR2620


define void @t2(i32 %m_task_id, i32 %start_x, i32 %end_x) nounwind {
	%A = fcmp olt <2 x double> zeroinitializer, zeroinitializer		; <<2 x i64>>:1 [#uses=1]
        sext <2 x i1> %A to <2 x i64>
	extractelement <2 x i64> %1, i32 1		; <i64>:2 [#uses=1]
	lshr i64 %2, 63		; <i64>:3 [#uses=1]
	trunc i64 %3 to i1		; <i1>:4 [#uses=1]
	zext i1 %4 to i8		; <i8>:5 [#uses=1]
	insertelement <2 x i8> zeroinitializer, i8 %5, i32 1		; <<2 x i8>>:6 [#uses=1]
	store <2 x i8> %6, <2 x i8>* null
	ret void
}
