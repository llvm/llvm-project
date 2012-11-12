; RUN: llc < %s -march=arm -mattr=+v4t | not grep mov

define i32 @f1() {
	%buf = alloca [32 x i32], align 4
	%tmp = getelementptr [32 x i32]* %buf, i32 0, i32 0
	%tmp1 = load i32* %tmp
	ret i32 %tmp1
}

define i32 @f2() {
	%buf = alloca [32 x i8], align 4
	%tmp = getelementptr [32 x i8]* %buf, i32 0, i32 0
	%tmp1 = load i8* %tmp
        %tmp2 = zext i8 %tmp1 to i32
	ret i32 %tmp2
}

define i32 @f3() {
	%buf = alloca [32 x i32], align 4
	%tmp = getelementptr [32 x i32]* %buf, i32 0, i32 32
	%tmp1 = load i32* %tmp
	ret i32 %tmp1
}

define i32 @f4() {
	%buf = alloca [32 x i8], align 4
	%tmp = getelementptr [32 x i8]* %buf, i32 0, i32 2
	%tmp1 = load i8* %tmp
        %tmp2 = zext i8 %tmp1 to i32
	ret i32 %tmp2
}
