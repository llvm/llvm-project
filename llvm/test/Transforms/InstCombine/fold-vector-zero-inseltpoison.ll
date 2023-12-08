; RUN: opt < %s -passes=instcombine -S | not grep zeroinitializer

define void @foo(i64 %A, i64 %B) {
bb8:
	br label %bb30

bb30:
	%s0 = phi i64 [ 0, %bb8 ], [ %r21, %bb30 ]
	%l0 = phi i64 [ -2222, %bb8 ], [ %r23, %bb30 ]
	%r2 = add i64 %s0, %B
	%r3 = inttoptr i64 %r2 to ptr
	%r4 = load <2 x double>, ptr %r3, align 8
	%r6 = bitcast <2 x double> %r4 to <2 x i64>
	%r7 = bitcast <2 x double> zeroinitializer to <2 x i64>
	%r8 = insertelement <2 x i64> poison, i64 9223372036854775807, i32 0
	%r9 = insertelement <2 x i64> poison, i64 -9223372036854775808, i32 0
	%r10 = insertelement <2 x i64> %r8, i64 9223372036854775807, i32 1
	%r11 = insertelement <2 x i64> %r9, i64 -9223372036854775808, i32 1
	%r12 = and <2 x i64> %r6, %r10
	%r13 = and <2 x i64> %r7, %r11
	%r14 = or <2 x i64> %r12, %r13
	%r15 = bitcast <2 x i64> %r14 to <2 x double>
	%r18 = add i64 %s0, %A
	%r19 = inttoptr i64 %r18 to ptr
	store <2 x double> %r15, ptr %r19, align 8
	%r21 = add i64 16, %s0
	%r23 = add i64 1, %l0
	%r25 = icmp slt i64 %r23, 0
	%r26 = zext i1 %r25 to i64
	%r27 = icmp ne i64 %r26, 0
	br i1 %r27, label %bb30, label %bb5

bb5:
	ret void
}
