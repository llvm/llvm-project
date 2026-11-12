; RUN: llc < %s -mtriple=i686--

@data = external global [339 x i64]

define void @foo(...) {
bb1:
	%t43 = load i64, ptr getelementptr ([339 x i64], ptr @data, i32 0, i64 212), align 4
	br i1 false, label %bb80, label %bb6
bb6:
	br i1 false, label %bb38, label %bb265
bb265:
	ret void
bb38:
	br i1 false, label %bb80, label %bb49
bb80:
	br i1 false, label %bb146, label %bb268
bb49:
	ret void
bb113:
	ret void
bb268:
	%t1062 = shl i64 %t43, 3
	%t1066 = shl i64 0, 3
	br label %bb85
bb85:
	%t1025 = phi i64 [ 0, %bb268 ], [ %t102.0, %bb234 ]
	%t1028 = phi i64 [ 0, %bb268 ], [ %t1066, %bb234 ]
	%t1031 = phi i64 [ 0, %bb268 ], [ %t103.0, %bb234 ]
	%t1034 = phi i64 [ 0, %bb268 ], [ %t1066, %bb234 ]
	%t102.0 = add i64 %t1028, %t1025
	%t103.0 = add i64 %t1034, %t1031
	br label %bb86
bb86:
	%t108.0 = phi i64 [ %t102.0, %bb85 ], [ %t1139, %bb248 ]
	%t110.0 = phi i64 [ %t103.0, %bb85 ], [ %t1142, %bb248 ]
	br label %bb193
bb193:
	%t1081 = add i64 %t110.0, -8
	%t1087 = add i64 %t108.0, -8
	br i1 false, label %bb193, label %bb248
bb248:
	%t1139 = add i64 %t108.0, %t1062
	%t1142 = add i64 %t110.0, %t1062
	br i1 false, label %bb86, label %bb234
bb234:
	br i1 false, label %bb85, label %bb113
bb146:
	ret void
}
