; RUN: llc < %s -march=arm | grep ldrb.*7 | count 1
; RUN: llc < %s -march=arm | grep ldrsb.*7 | count 1

	%struct.A = type { i8, i8, i8, i8, i16, i8, i8, %struct.B** }
	%struct.B = type { float, float, i32, i32, i32, [0 x i8] }

define i8 @f1(%struct.A* %d) {
	%tmp2 = getelementptr %struct.A* %d, i32 0, i32 4
	%tmp23 = bitcast i16* %tmp2 to i32*
	%tmp4 = load i32* %tmp23
	%tmp512 = lshr i32 %tmp4, 24
	%tmp56 = trunc i32 %tmp512 to i8
	ret i8 %tmp56
}

define i32 @f2(%struct.A* %d) {
	%tmp2 = getelementptr %struct.A* %d, i32 0, i32 4
	%tmp23 = bitcast i16* %tmp2 to i32*
	%tmp4 = load i32* %tmp23
	%tmp512 = lshr i32 %tmp4, 24
	%tmp56 = trunc i32 %tmp512 to i8
        %tmp57 = sext i8 %tmp56 to i32
	ret i32 %tmp57
}
