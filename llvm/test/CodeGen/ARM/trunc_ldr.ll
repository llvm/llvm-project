; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

	%struct.A = type { i8, i8, i8, i8, i16, i8, i8, ptr }
	%struct.B = type { float, float, i32, i32, i32, [0 x i8] }

define i8 @f1(ptr %d) {
	%tmp2 = getelementptr %struct.A, ptr %d, i32 0, i32 4
	%tmp4 = load i32, ptr %tmp2
	%tmp512 = lshr i32 %tmp4, 24
	%tmp56 = trunc i32 %tmp512 to i8
	ret i8 %tmp56
}

define i32 @f2(ptr %d) {
	%tmp2 = getelementptr %struct.A, ptr %d, i32 0, i32 4
	%tmp4 = load i32, ptr %tmp2
	%tmp512 = lshr i32 %tmp4, 24
	%tmp56 = trunc i32 %tmp512 to i8
        %tmp57 = sext i8 %tmp56 to i32
	ret i32 %tmp57
}

; CHECK: ldrb{{.*}}7
; CHECK-NOT: ldrb{{.*}}7

; CHECK: ldrsb{{.*}}7
; CHECK-NOT: ldrsb{{.*}}7

